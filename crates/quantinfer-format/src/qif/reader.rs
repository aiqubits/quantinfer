use super::*;

pub(crate) fn has_magic(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

pub(crate) fn uses_ggml_dimension_order(metadata: &BTreeMap<String, GgufValue>) -> Result<bool> {
    match metadata.get(GGML_DIMENSION_ORDER) {
        None => Ok(false),
        Some(GgufValue::Bool(value)) => Ok(*value),
        Some(_) => Err(invalid("GGML dimension order metadata must be a boolean")),
    }
}

pub(crate) fn open_packed(bytes: Vec<u8>) -> Result<PackedTensorReader> {
    let QifReader {
        manifest,
        metadata,
        tensors,
        ..
    } = QifReader::open(&bytes)?;
    if metadata.get("general.architecture").is_some_and(|value| {
        !matches!(value, GgufValue::String(architecture) if architecture == &manifest.architecture)
    }) {
        return Err(invalid("metadata architecture disagrees with the manifest"));
    }
    let ggml = uses_ggml_dimension_order(&metadata)?;
    let tensors = manifest
        .tensors
        .iter()
        .map(|tensor| {
            let range = &tensors[&tensor.name];
            let mut shape = tensor.descriptor.logical_shape.dims().to_vec();
            if ggml {
                shape.reverse();
            }
            GgufTensorInfo {
                name: tensor.name.clone(),
                shape,
                dtype: storage_encoding(tensor.descriptor.dtype).0,
                offset: range.start as u64,
                size: range.len(),
            }
        })
        .collect();
    Ok(PackedTensorReader {
        bytes,
        metadata,
        tensors,
        data_start: 0,
        embedded_manifest: Some(manifest),
    })
}

/// A validated QIF artifact. Tensor payloads borrow the original bytes.
pub struct QifReader<'a> {
    bytes: &'a [u8],
    manifest: ModelManifest,
    metadata: BTreeMap<String, GgufValue>,
    pub(crate) tensors: BTreeMap<String, Range<usize>>,
}

impl<'a> QifReader<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let (prefix, body) = bytes
            .split_at_checked(PREFIX)
            .ok_or_else(|| invalid("truncated prefix"))?;
        if &prefix[..4] != MAGIC {
            return Err(invalid("invalid magic"));
        }
        let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(invalid(format!("unsupported version {version}")));
        }
        let manifest_size = header_size(&prefix[8..16])?;
        let metadata_size = header_size(&prefix[16..24])?;
        let header_size = manifest_size + metadata_size;
        if header_size > MAX_HEADER_BYTES {
            return Err(invalid("header exceeds the size limit"));
        }
        let header = body
            .get(..header_size)
            .ok_or_else(|| invalid("truncated header"))?;
        if Sha256::digest(header)[..] != prefix[24..] {
            return Err(invalid("header checksum mismatch"));
        }
        let (manifest, metadata) = header.split_at(manifest_size);
        let manifest: ModelManifest = serde_json::from_slice(manifest)
            .map_err(|error| invalid(format!("invalid manifest: {error}")))?;
        manifest.validate()?;
        let metadata = read_metadata(metadata)
            .map_err(|error| invalid(format!("invalid metadata: {error}")))?;

        let mut tensors = BTreeMap::new();
        let mut end = PREFIX + header_size;
        for tensor in &manifest.tensors {
            let start = aligned(end)?;
            check_padding(bytes, end..start)?;
            end = start
                .checked_add(payload_size(&tensor.descriptor)?)
                .ok_or_else(|| invalid("tensor end overflows usize"))?;
            let payload = bytes
                .get(start..end)
                .ok_or_else(|| invalid(format!("tensor {} is truncated", tensor.name)))?;
            if tensor.checksum_sha256.as_deref() != Some(&sha256_hex(payload)) {
                return Err(invalid(format!("tensor {} checksum mismatch", tensor.name)));
            }
            tensors.insert(tensor.name.clone(), start..end);
        }
        if end != bytes.len() {
            return Err(invalid("unexpected bytes after the last tensor"));
        }
        Ok(Self {
            bytes,
            manifest,
            metadata,
            tensors,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&'a [u8]> {
        self.tensors
            .get(name)
            .map(|range| &self.bytes[range.clone()])
            .ok_or_else(|| invalid(format!("unknown tensor {name}")))
    }
}
