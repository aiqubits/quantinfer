//! Versioned QuantInfer model containers.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write;
use std::ops::Range;

use quantinfer_core::{
    DType, QuantizationScheme, Shape, TensorStorageDescriptor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
};
use sha2::{Digest, Sha256};

use crate::{
    encode_value, sha256_hex, tensor_byte_size, write_string, Cursor, FormatError, GgmlType,
    GgufReader, GgufTensorInfo, GgufValue, ModelManifest, PackedTensorReader, Result,
    TensorManifest,
};

const MAGIC: &[u8; 4] = b"QIF\0";
const VERSION: u32 = 1;
const PREFIX: usize = 56;
const ALIGNMENT: usize = 64;
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARRAY_ITEMS: usize = 1_048_576;
const GGML_DIMENSION_ORDER: &str = "quantinfer.qif.ggml_dimension_order";

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
    if ggml && manifest.architecture != "llama" {
        return Err(invalid("GGML dimension order requires a Llama model"));
    }
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

/// Transcode a supported GGUF model without dequantizing or requantizing.
pub fn import_gguf(
    reader: &GgufReader,
    model_id: impl Into<String>,
    output: &mut impl Write,
) -> Result<()> {
    let mut metadata = reader.metadata().clone();
    let architecture = match metadata.get("general.architecture") {
        Some(GgufValue::String(architecture)) => architecture.as_str(),
        None => "qwen3",
        Some(_) => return Err(invalid("source architecture must be a string")),
    };
    let mut manifest = ModelManifest::new(model_id, architecture);
    let ggml = architecture == "llama" && reader.llama_uses_ggml_layout()?;
    metadata.insert(GGML_DIMENSION_ORDER.to_owned(), GgufValue::Bool(ggml));
    let mut payloads = BTreeMap::new();
    let mut schemes = Vec::new();
    for tensor in reader.tensors() {
        let bytes = reader.tensor_bytes(&tensor.name)?;
        let (dtype, scheme, block_size) = match tensor.dtype {
            GgmlType::F32 => (DType::F32, QuantizationScheme::None, 0),
            GgmlType::F16 => (DType::F16, QuantizationScheme::None, 0),
            GgmlType::BF16 => (DType::BF16, QuantizationScheme::None, 0),
            GgmlType::Q8_0 => (DType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
            GgmlType::Q4_0 => (DType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
            GgmlType::Q4_K => (DType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q5_K => (DType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q6_K => (DType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
        };
        let mut shape = tensor.shape.clone();
        if ggml {
            shape.reverse();
        }
        let shape = Shape::new(shape)?;
        let descriptor = if scheme == QuantizationScheme::None {
            TensorStorageDescriptor::floating(dtype, shape)?
        } else {
            if !schemes.contains(&scheme) {
                schemes.push(scheme);
            }
            TensorStorageDescriptor::quantized(dtype, scheme, shape, block_size)?
        };
        let payload = if dtype == DType::Q4_0 && reader.uses_legacy_quant_block_layout()? {
            Cow::Owned(repack_q4_0(bytes))
        } else {
            Cow::Borrowed(bytes)
        };
        payloads.insert(tensor.name.as_str(), payload);
        manifest.tensors.push(TensorManifest {
            name: tensor.name.clone(),
            descriptor,
            checksum_sha256: None,
        });
    }
    manifest.quantization = if matches!(
        metadata.get("quantinfer.quantization.variant"),
        Some(GgufValue::String(variant)) if variant == "mixed"
    ) {
        QuantizationScheme::Mixed
    } else {
        match schemes.as_slice() {
            [] => QuantizationScheme::None,
            [scheme] => *scheme,
            _ => QuantizationScheme::Mixed,
        }
    };
    write_qif(
        manifest,
        &metadata,
        |name| Ok(payloads[name].as_ref()),
        output,
    )
}

fn repack_q4_0(bytes: &[u8]) -> Vec<u8> {
    const PACKED_BYTES: usize = Q4_0_BLOCK_SIZE / 2;
    const BLOCK_BYTES: usize = 2 + PACKED_BYTES;
    let mut result = bytes.to_vec();
    for (source, target) in bytes
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(result.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        let pairs = &source[2..];
        for (index, byte) in target[2..].iter_mut().enumerate() {
            let shift = (index % 2) * 4;
            let low = (pairs[index / 2] >> shift) & 0x0f;
            let high = (pairs[index / 2 + PACKED_BYTES / 2] >> shift) & 0x0f;
            *byte = low | (high << 4);
        }
    }
    result
}

/// A validated QIF artifact. Tensor payloads borrow the original bytes.
pub struct QifReader<'a> {
    bytes: &'a [u8],
    manifest: ModelManifest,
    metadata: BTreeMap<String, GgufValue>,
    tensors: BTreeMap<String, Range<usize>>,
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

/// Write a QIF artifact to a stream. The manifest provides tensor order and
/// descriptors, and the writer computes the SHA-256 for each payload.
pub fn write_qif<'a>(
    mut manifest: ModelManifest,
    metadata: &BTreeMap<String, GgufValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
) -> Result<()> {
    manifest.validate()?;
    let mut payloads = Vec::with_capacity(manifest.tensors.len());
    for tensor in &mut manifest.tensors {
        let payload = tensor_bytes(&tensor.name)?;
        if payload.len() != payload_size(&tensor.descriptor)? {
            return Err(invalid(format!(
                "tensor {} has an invalid length",
                tensor.name
            )));
        }
        tensor.checksum_sha256 = Some(sha256_hex(payload));
        payloads.push(payload);
    }
    let manifest = serde_json::to_vec(&manifest)
        .map_err(|error| invalid(format!("cannot serialize manifest: {error}")))?;
    let metadata = write_metadata(metadata)?;
    let header_size = manifest.len().saturating_add(metadata.len());
    if header_size > MAX_HEADER_BYTES {
        return Err(invalid("header exceeds the size limit"));
    }
    let mut digest = Sha256::new();
    digest.update(&manifest);
    digest.update(&metadata);
    let checksum = digest.finalize();

    let mut end = PREFIX + header_size;
    output
        .write_all(MAGIC)
        .and_then(|()| output.write_all(&VERSION.to_le_bytes()))
        .and_then(|()| output.write_all(&(manifest.len() as u64).to_le_bytes()))
        .and_then(|()| output.write_all(&(metadata.len() as u64).to_le_bytes()))
        .and_then(|()| output.write_all(&checksum))
        .and_then(|()| output.write_all(&manifest))
        .and_then(|()| output.write_all(&metadata))
        .map_err(write_error)?;
    for payload in payloads {
        let start = aligned(end)?;
        let padding = start - end;
        end = start
            .checked_add(payload.len())
            .ok_or_else(|| invalid("tensor end overflows usize"))?;
        output
            .write_all(&[0; ALIGNMENT][..padding])
            .and_then(|()| output.write_all(payload))
            .map_err(write_error)?;
    }
    Ok(())
}

fn payload_size(descriptor: &TensorStorageDescriptor) -> Result<usize> {
    let (encoding, scheme, block_size) = storage_encoding(descriptor.dtype);
    let shape = descriptor.logical_shape.clone();
    let expected = if scheme == QuantizationScheme::None {
        TensorStorageDescriptor::floating(descriptor.dtype, shape)?
    } else {
        TensorStorageDescriptor::quantized(descriptor.dtype, scheme, shape, block_size)?
    };
    if descriptor != &expected {
        return Err(invalid("unsupported tensor storage descriptor"));
    }
    tensor_byte_size(encoding, descriptor.logical_shape.dims())
}

fn storage_encoding(dtype: DType) -> (GgmlType, QuantizationScheme, usize) {
    match dtype {
        DType::F32 => (GgmlType::F32, QuantizationScheme::None, 0),
        DType::F16 => (GgmlType::F16, QuantizationScheme::None, 0),
        DType::BF16 => (GgmlType::BF16, QuantizationScheme::None, 0),
        DType::Q8_0 => (GgmlType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
        DType::Q4_0 => (GgmlType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
        DType::Q4_K => (GgmlType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
        DType::Q5_K => (GgmlType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
        DType::Q6_K => (GgmlType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
    }
}

fn read_metadata(bytes: &[u8]) -> Result<BTreeMap<String, GgufValue>> {
    let mut cursor = Cursor::new(bytes);
    let count = cursor.read_u64()?;
    if count > (bytes.len() / 12) as u64 {
        return Err(invalid("metadata count exceeds its section"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = cursor.read_string()?;
        let value = cursor.read_value_with_limit(MAX_ARRAY_ITEMS)?;
        if key.is_empty() || metadata.insert(key, value).is_some() {
            return Err(invalid("metadata keys must be unique and nonempty"));
        }
    }
    if cursor.position() != bytes.len() {
        return Err(invalid("unexpected bytes after metadata"));
    }
    Ok(metadata)
}

fn write_metadata(metadata: &BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    for (key, value) in metadata {
        if key.is_empty() {
            return Err(invalid("metadata keys must be nonempty"));
        }
        if matches!(value, GgufValue::Array(values) if values.len() > MAX_ARRAY_ITEMS) {
            return Err(invalid("metadata array exceeds the item limit"));
        }
        write_string(&mut bytes, key)?;
        if matches!(value, GgufValue::Array(values) if values.is_empty()) {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
        } else {
            encode_value(&mut bytes, value)?;
        }
    }
    Ok(bytes)
}

fn header_size(bytes: &[u8]) -> Result<usize> {
    let size = u64::from_le_bytes(bytes.try_into().unwrap());
    if size > MAX_HEADER_BYTES as u64 {
        return Err(invalid("header section exceeds the size limit"));
    }
    Ok(size as usize)
}

fn aligned(offset: usize) -> Result<usize> {
    offset
        .checked_next_multiple_of(ALIGNMENT)
        .ok_or_else(|| invalid("tensor alignment overflows usize"))
}

fn check_padding(bytes: &[u8], range: Range<usize>) -> Result<()> {
    if !bytes
        .get(range)
        .is_some_and(|padding| padding.iter().all(|&byte| byte == 0))
    {
        return Err(invalid("missing or nonzero tensor padding"));
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> FormatError {
    FormatError::Invalid(format!("QIF {message}"))
}

fn write_error(error: std::io::Error) -> FormatError {
    FormatError::Io(format!("cannot write QIF: {error}"))
}

#[cfg(test)]
mod tests {
    use quantinfer_core::{Q4_0Block, QuantizedTensor4};

    use super::*;
    use crate::{GgufWriter, QUANT_BLOCK_LAYOUT};

    const WEIGHT: [u8; 18] = [
        0, 60, 0x91, 0x82, 0x73, 0x64, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
    ];
    const NORM: [u8; 4] = 1.5f32.to_le_bytes();

    fn model() -> (ModelManifest, BTreeMap<String, GgufValue>) {
        let mut manifest = ModelManifest::new("qif-fixture", "llama");
        manifest.quantization = QuantizationScheme::Q4_0;
        manifest.tensors = vec![
            TensorManifest {
                name: "norm".to_owned(),
                descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(1)).unwrap(),
                checksum_sha256: None,
            },
            TensorManifest {
                name: "weight".to_owned(),
                descriptor: TensorStorageDescriptor::quantized(
                    DType::Q4_0,
                    QuantizationScheme::Q4_0,
                    Shape::from(vec![1, 32]),
                    Q4_0_BLOCK_SIZE,
                )
                .unwrap(),
                checksum_sha256: None,
            },
        ];
        let metadata = BTreeMap::from([
            ("context".to_owned(), GgufValue::U64(4096)),
            ("empty".to_owned(), GgufValue::Array(Vec::new())),
            (
                "tokens".to_owned(),
                GgufValue::Array(vec![GgufValue::String("量化".to_owned())]),
            ),
        ]);
        (manifest, metadata)
    }

    fn artifact() -> Vec<u8> {
        let (manifest, metadata) = model();
        let mut bytes = Vec::new();
        write_qif(
            manifest,
            &metadata,
            |name| Ok(if name == "norm" { &NORM[..] } else { &WEIGHT }),
            &mut bytes,
        )
        .unwrap();
        bytes
    }

    #[test]
    fn round_trips_manifest_metadata_and_aligned_payloads() {
        let bytes = artifact();
        assert_eq!(&bytes[..8], b"QIF\0\x01\0\0\0");
        let reader = QifReader::open(&bytes).unwrap();
        let (manifest, metadata) = model();
        assert_eq!(reader.manifest().architecture, manifest.architecture);
        assert_eq!(reader.metadata(), &metadata);
        assert_eq!(reader.tensor_bytes("norm").unwrap(), NORM);
        assert_eq!(reader.tensor_bytes("weight").unwrap(), WEIGHT);
        assert!(reader.tensor_bytes("unknown").is_err());
        assert!(reader
            .tensors
            .values()
            .all(|range| range.start % ALIGNMENT == 0));
        assert_eq!(
            reader.manifest().tensors[1].checksum_sha256.as_deref(),
            Some(sha256_hex(&WEIGHT).as_str()),
        );
    }

    #[test]
    fn rejects_unsupported_versions_and_damaged_artifacts() {
        let bytes = artifact();
        let reader = QifReader::open(&bytes).unwrap();
        let damaged_offsets = [
            (4, "unsupported version"),
            (PREFIX, "header checksum mismatch"),
            (reader.tensors["norm"].end, "nonzero tensor padding"),
            (
                reader.tensors["weight"].start,
                "tensor weight checksum mismatch",
            ),
        ];
        for (offset, message) in damaged_offsets {
            let mut damaged = bytes.clone();
            damaged[offset] ^= 1;
            let error = QifReader::open(&damaged)
                .err()
                .expect("damage was accepted");
            assert!(error.to_string().contains(message), "{error}");
        }
        assert!(QifReader::open(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(QifReader::open(&trailing).is_err());
    }

    #[test]
    fn validates_all_descriptors_and_lengths_before_writing() {
        let (mut manifest, metadata) = model();
        let mut bytes = Vec::new();
        assert!(write_qif(manifest.clone(), &metadata, |_| Ok(&NORM), &mut bytes).is_err());
        assert!(bytes.is_empty());
        manifest.tensors[1].descriptor.logical_shape = Shape::from(vec![usize::MAX, 32]);
        assert!(write_qif(manifest, &metadata, |_| Ok(&NORM), &mut bytes).is_err());
        assert!(bytes.is_empty());
    }

    #[test]
    fn imports_legacy_q4_0_without_changing_decoded_values() {
        let block = Q4_0Block {
            scale: 0.5,
            values: std::array::from_fn(|index| ((index * 7 + index / 4) % 16) as u8),
        };
        let tensor = QuantizedTensor4::from_blocks(vec![1, 32], vec![block.clone()]).unwrap();
        let mut writer = GgufWriter::new();
        writer.add_q4_0_tensor("weight", &tensor);
        writer.metadata.remove(QUANT_BLOCK_LAYOUT);
        for (index, pair) in block.values.as_chunks::<2>().0.iter().enumerate() {
            writer.tensors[0].data[index + 2] = pair[0] | (pair[1] << 4);
        }
        let gguf = GgufReader::open(writer.finish().unwrap()).unwrap();
        assert_eq!(gguf.tensor_q4_0("weight").unwrap(), tensor);
        let mut bytes = Vec::new();
        import_gguf(&gguf, "legacy", &mut bytes).unwrap();
        let qif = PackedTensorReader::open_artifact(bytes).unwrap();
        assert_ne!(
            gguf.tensor_bytes("weight").unwrap(),
            qif.tensor_bytes("weight").unwrap()
        );
        assert_eq!(qif.tensor_q4_0("weight").unwrap(), tensor);
    }
}
