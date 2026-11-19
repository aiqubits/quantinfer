//! GGUF header and tensor payload index reader.

use std::collections::{BTreeMap, BTreeSet};
use std::{borrow::Cow, path::Path, sync::Arc};

use crate::{manifest::ModelManifest, qif_v2};

use super::encoding::{align_up, tensor_byte_size, Cursor};
use super::types::{FormatError, GgmlType, GgufTensorInfo, GgufValue};
use super::{
    Result, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, MAX_GGUF_HEADER_BYTES,
    MAX_GGUF_TENSOR_RANK, QUANT_BLOCK_LAYOUT,
};

/// Indexed supported encodings with owned or on-demand file-backed payloads.
#[derive(Debug, Clone)]
pub struct PackedTensorReader {
    pub(crate) bytes: Vec<u8>,
    pub(crate) metadata: BTreeMap<String, GgufValue>,
    pub(crate) tensors: Vec<GgufTensorInfo>,
    pub(crate) data_start: usize,
    pub(crate) embedded_manifest: Option<ModelManifest>,
    pub(crate) qif_metadata: Option<BTreeMap<String, qif_v2::MetadataValue>>,
    pub(crate) file_backing: Option<Arc<qif_v2::FileBacking>>,
}

pub type GgufReader = PackedTensorReader;

impl PackedTensorReader {
    /// Open native QIF SINGLE/PACKAGE with control-only I/O and on-demand tensor
    /// reads. GGUF retains its existing owned reader strategy.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_path_with_memory_budget(path, None)
    }

    /// Open a caller-owned descriptor with the same QIF on-demand strategy.
    pub fn open_file_with_memory_budget(
        file: std::fs::File,
        budget: Option<usize>,
    ) -> Result<Self> {
        qif_v2::open_packed_file(file, budget)
    }

    /// Preflight QIF physical weights and controls before reading tensor data.
    /// The budget is a storage lower bound, not an allocator/workspace limit.
    pub fn open_path_with_memory_budget(
        path: impl AsRef<Path>,
        budget: Option<usize>,
    ) -> Result<Self> {
        qif_v2::open_packed_path(path.as_ref(), budget)
    }

    /// Open GGUF or QIF based on its file magic.
    pub fn open_artifact(bytes: Vec<u8>) -> Result<Self> {
        Self::open_artifact_with_memory_budget(bytes, None)
    }

    /// Reject artifacts whose input plus required QIF materialized weights
    /// exceed the budget before decoding. This is a storage lower-bound check,
    /// not a bound on allocator overhead, decoder workspace or runtime state.
    pub fn open_artifact_with_memory_budget(bytes: Vec<u8>, budget: Option<usize>) -> Result<Self> {
        if budget.is_some_and(|limit| bytes.len() > limit) {
            return Err(FormatError::Invalid(
                "model input exceeds memory budget".to_owned(),
            ));
        }
        if qif_v2::has_magic(&bytes) {
            qif_v2::open_packed(bytes, budget)
        } else {
            Self::open(bytes)
        }
    }

    /// Check each declared tensor against this artifact's dtype and model shape view.
    /// Runtime feature/version policy is checked by the runtime separately.
    pub fn validate_manifest(&self, manifest: &ModelManifest) -> Result<()> {
        use quantinfer_core::DType;
        manifest.validate()?;
        let ggml_layout = self.uses_ggml_output_rows()?;
        for tensor in &manifest.tensors {
            let info = self.tensor_info(&tensor.name).map_err(|_| {
                FormatError::Invalid(format!(
                    "model manifest tensor {} is missing from the tensor file",
                    tensor.name
                ))
            })?;
            let expected_dtype = match tensor.descriptor.dtype {
                DType::F32 => GgmlType::F32,
                DType::F16 => GgmlType::F16,
                DType::BF16 => GgmlType::BF16,
                DType::Q8_0 => GgmlType::Q8_0,
                DType::Q4_0 => GgmlType::Q4_0,
                DType::Q4_K => GgmlType::Q4_K,
                DType::Q5_K => GgmlType::Q5_K,
                DType::Q6_K => GgmlType::Q6_K,
            };
            if info.dtype != expected_dtype {
                return Err(FormatError::Invalid(format!(
                    "model manifest tensor {} declares {:?}, but the tensor file stores {:?}",
                    tensor.name, expected_dtype, info.dtype
                )));
            }
            let declared_shape = tensor.descriptor.logical_shape.dims();
            let shape_matches = if ggml_layout
                && (self.embedded_manifest().is_some() || tensor.name == "token_embd.weight")
            {
                if declared_shape.len() == 2 {
                    info.shape.as_slice() == [declared_shape[1], declared_shape[0]]
                } else {
                    info.shape.as_slice() == declared_shape
                }
            } else {
                info.shape.as_slice() == declared_shape
            };
            if !shape_matches {
                return Err(FormatError::Invalid(format!(
                    "model manifest tensor {} declares shape {:?}, but the tensor file stores {:?}",
                    tensor.name, declared_shape, info.shape
                )));
            }
            let recorded = self.tensor_storage_layout(&tensor.name)?;
            let declared = match tensor.descriptor.physical_layout.as_str() {
                "output_rows" => super::GGML_OUTPUT_ROWS_LAYOUT,
                value => value,
            };
            if declared != recorded {
                return Err(FormatError::Invalid(format!(
                    "model manifest tensor {} declares layout {declared:?}, but the artifact stores {recorded:?}",
                    tensor.name
                )));
            }
        }
        Ok(())
    }

    pub fn embedded_manifest(&self) -> Option<&ModelManifest> {
        self.embedded_manifest.as_ref()
    }

    /// The complete native metadata when opened from QIF. The GGUF-compatible
    /// `metadata()` view omits entries containing maps, which GGUF cannot encode.
    pub fn qif_metadata(&self) -> Option<&BTreeMap<String, crate::qif::MetadataValue>> {
        self.qif_metadata.as_ref()
    }

    /// Detect GGML dimension order and packed projection rows from generic
    /// GGUF metadata. Architecture-specific interpretation belongs to model
    /// adapters, not the format reader.
    pub fn uses_ggml_output_rows(&self) -> Result<bool> {
        if self.embedded_manifest.is_some() {
            return match self.metadata.get(qif_v2::GGML_DIMENSION_ORDER) {
                Some(GgufValue::Bool(value)) => Ok(*value),
                None => Ok(false),
                Some(_) => Err(FormatError::Invalid(
                    "QIF GGML dimension order metadata must be a boolean".into(),
                )),
            };
        }
        gguf_uses_ggml_rows(&self.metadata, &self.tensors)
    }

    pub(crate) fn uses_legacy_quant_block_layout(&self) -> Result<bool> {
        if self.embedded_manifest.is_some() {
            return Ok(false);
        }
        gguf_uses_legacy_blocks(&self.metadata)
    }

    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        let GgufControls {
            metadata,
            tensors,
            data_start,
        } = Self::parse_gguf_controls(&bytes, bytes.len())?;
        Ok(Self {
            bytes,
            metadata,
            tensors,
            data_start,
            embedded_manifest: None,
            qif_metadata: None,
            file_backing: None,
        })
    }

    pub(super) fn parse_gguf_controls(bytes: &[u8], file_length: usize) -> Result<GgufControls> {
        let mut cursor = Cursor::new(bytes);
        if cursor.read_exact(4)? != &GGUF_MAGIC[..] {
            return Err(FormatError::Invalid("invalid GGUF magic".to_owned()));
        }
        if cursor.read_u32()? != GGUF_VERSION {
            return Err(FormatError::Invalid("unsupported GGUF version".to_owned()));
        }
        let tensor_count = usize::try_from(cursor.read_u64()?)
            .map_err(|_| FormatError::Invalid("GGUF tensor count overflows usize".to_owned()))?;
        let metadata_count = usize::try_from(cursor.read_u64()?)
            .map_err(|_| FormatError::Invalid("GGUF metadata count overflows usize".to_owned()))?;
        if tensor_count > bytes.len() / 16 || metadata_count > bytes.len() / 9 {
            return Err(FormatError::Invalid(
                "GGUF header counts exceed file capacity".to_owned(),
            ));
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            if key.is_empty() || metadata.insert(key, cursor.read_value()?).is_some() {
                return Err(FormatError::Invalid(
                    "GGUF metadata keys must be unique and non-empty".to_owned(),
                ));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(value)) => *value as u64,
            Some(GgufValue::U64(value)) => *value,
            _ => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(FormatError::Invalid(
                "GGUF alignment must be a non-zero power of two".to_owned(),
            ));
        }
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            if name.is_empty() {
                return Err(FormatError::Invalid(
                    "GGUF tensor names must be non-empty".to_owned(),
                ));
            }
            if !tensor_names.insert(name.clone()) {
                return Err(FormatError::Invalid(format!(
                    "duplicate GGUF tensor name {name}"
                )));
            }
            let dimensions = cursor.read_u32()? as usize;
            let remaining = bytes.len().saturating_sub(cursor.position());
            if dimensions > MAX_GGUF_TENSOR_RANK || dimensions > remaining / 8 {
                return Err(FormatError::Invalid(
                    "GGUF tensor rank exceeds the supported header capacity".to_owned(),
                ));
            }
            let mut shape = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                shape.push(usize::try_from(cursor.read_u64()?).map_err(|_| {
                    FormatError::Invalid("GGUF tensor dimension overflows usize".to_owned())
                })?);
            }
            if shape.contains(&0) {
                return Err(FormatError::Invalid(
                    "GGUF tensor dimensions must be greater than zero".to_owned(),
                ));
            }
            let dtype = GgmlType::try_from(cursor.read_u32()?)?;
            let offset = cursor.read_u64()?;
            let size = tensor_byte_size(dtype, &shape)?;
            tensors.push(GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
                size,
            });
        }
        if cursor.position() > MAX_GGUF_HEADER_BYTES {
            return Err(FormatError::Invalid(
                "GGUF header exceeds the size limit".to_owned(),
            ));
        }
        let data_start = usize::try_from(align_up(cursor.position() as u64, alignment)?)
            .map_err(|_| FormatError::Invalid("GGUF data offset overflows usize".to_owned()))?;
        let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(tensors.len());
        for tensor in &tensors {
            if !tensor.offset.is_multiple_of(alignment) {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {} offset is not aligned",
                    tensor.name
                )));
            }
            let end = tensor
                .offset
                .checked_add(u64::try_from(tensor.size).map_err(|_| {
                    FormatError::Invalid("GGUF tensor size overflows u64".to_owned())
                })?)
                .ok_or_else(|| {
                    FormatError::Invalid("GGUF tensor range overflows u64".to_owned())
                })?;
            let data_end = data_start
                .checked_add(usize::try_from(tensor.offset).map_err(|_| {
                    FormatError::Invalid("GGUF tensor offset overflows usize".to_owned())
                })?)
                .and_then(|start| start.checked_add(tensor.size))
                .ok_or_else(|| {
                    FormatError::Invalid(format!(
                        "GGUF tensor {} range overflows the file",
                        tensor.name
                    ))
                })?;
            if data_end > file_length {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {} data is truncated",
                    tensor.name
                )));
            }
            ranges.push((tensor.offset, end));
        }
        ranges.sort_unstable_by_key(|(start, _)| *start);
        if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
            return Err(FormatError::Invalid(
                "GGUF tensor payloads overlap".to_owned(),
            ));
        }
        Ok(GgufControls {
            metadata,
            tensors,
            data_start,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    /// Look up a parameter without treating an unprojectable native QIF value
    /// as absent. Model adapters should use this when applying defaults.
    pub fn metadata_value(&self, key: &str) -> Result<Option<&GgufValue>> {
        let value = self.metadata.get(key);
        if value.is_none()
            && self
                .qif_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.contains_key(key))
        {
            return Err(FormatError::Invalid(format!(
                "QIF metadata {key} contains a map, unsupported for this model parameter"
            )));
        }
        Ok(value)
    }

    pub fn file_size_bytes(&self) -> usize {
        self.file_backing
            .as_ref()
            .map_or(self.bytes.len(), |source| source.artifact_length)
    }

    pub fn is_file_backed(&self) -> bool {
        self.file_backing.is_some()
    }

    /// Bytes retained by explicit borrowed tensor requests; aliases count once.
    /// Typed tensor decoders use transient reads and do not populate this cache.
    pub fn cached_tensor_bytes(&self) -> usize {
        self.file_backing
            .as_ref()
            .map_or(0, |source| source.cached_bytes())
    }

    /// Serialized control size estimate plus owned/cached data and the largest
    /// transient tensor/chunk buffers. Decoder/allocator overhead is separate.
    pub fn loading_memory_bytes(&self) -> Result<usize> {
        match &self.file_backing {
            None => Ok(self.bytes.len()),
            Some(source) => source
                .control_length
                .checked_mul(2)
                .and_then(|value| value.checked_add(source.cached_bytes()))
                .and_then(|value| value.checked_add(source.loading_workspace))
                .ok_or_else(|| FormatError::Invalid("loading memory estimate overflows".into())),
        }
    }

    /// Full streaming QIF file/package verification without filling the cache.
    pub fn verify_qif(&self) -> Result<()> {
        self.file_backing
            .as_ref()
            .ok_or_else(|| {
                FormatError::Invalid("full QIF verification requires a file-backed reader".into())
            })?
            .verify()
    }

    /// Borrow owned payloads or read only the selected file-backed tensor.
    /// File-backed full-tensor reads check its raw hash and finite values.
    pub fn tensor_data(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        if let Some(source) = &self.file_backing {
            return source.data(name);
        }
        self.tensor_bytes(name).map(Cow::Borrowed)
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        if let Some(source) = &self.file_backing {
            return source.borrowed(name);
        }
        let info = self.tensor_info(name)?;
        let offset = usize::try_from(info.offset)
            .map_err(|_| FormatError::Invalid("GGUF tensor offset overflows usize".to_owned()))?;
        let start = self
            .data_start
            .checked_add(offset)
            .ok_or_else(|| FormatError::Invalid("GGUF tensor offset overflows usize".to_owned()))?;
        let end = start
            .checked_add(info.size)
            .ok_or_else(|| FormatError::Invalid("GGUF tensor size overflows usize".to_owned()))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| FormatError::Invalid("GGUF tensor data is truncated".to_owned()))
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown GGUF tensor {name}")))
    }
}

pub(super) fn gguf_uses_ggml_rows(
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[GgufTensorInfo],
) -> Result<bool> {
    if metadata
        .keys()
        .any(|key| key.ends_with(".tie_word_embeddings"))
    {
        return Ok(false);
    }
    let length = metadata
        .iter()
        .find(|(key, _)| key.ends_with(".embedding_length"));
    let Some(embedding) = tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
    else {
        return Ok(false);
    };
    if let (Some((length_key, value)), [first, second]) = (length, embedding.shape.as_slice()) {
        let hidden = match value {
            GgufValue::U32(value) => u64::from(*value),
            GgufValue::U64(value) => *value,
            GgufValue::I32(value) if *value >= 0 => *value as u64,
            GgufValue::I64(value) if *value >= 0 => *value as u64,
            _ => {
                return Err(FormatError::Invalid(format!(
                    "GGUF metadata {length_key} must be an unsigned integer"
                )))
            }
        };
        let hidden = usize::try_from(hidden).map_err(|_| {
            FormatError::Invalid(format!("GGUF metadata {length_key} overflows usize"))
        })?;
        match (*first == hidden, *second == hidden) {
            (true, false) => return Ok(true),
            (false, true) => return Ok(false),
            _ => {}
        }
    }
    Ok(metadata.contains_key("general.file_type"))
}

pub(super) fn gguf_uses_legacy_blocks(metadata: &BTreeMap<String, GgufValue>) -> Result<bool> {
    match metadata.get(QUANT_BLOCK_LAYOUT) {
        Some(GgufValue::String(layout)) if layout == "ggml" => Ok(false),
        Some(_) => Err(FormatError::Invalid(
            "unsupported GGUF quantization block layout".to_owned(),
        )),
        // Older QuantInfer converter outputs declare the architecture-specific
        // tie flag. The old standalone writer could omit the architecture.
        None => Ok(match metadata.get("general.architecture") {
            Some(GgufValue::String(architecture)) => {
                metadata.contains_key(&format!("{architecture}.tie_word_embeddings"))
            }
            _ => true,
        }),
    }
}

pub(crate) struct GgufControls {
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_start: usize,
}
