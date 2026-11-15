//! GGUF header and tensor payload index reader.

use std::collections::{BTreeMap, BTreeSet};

use crate::{manifest::ModelManifest, qif, qif_v2};

use super::encoding::{align_up, tensor_byte_size, Cursor};
use super::types::{FormatError, GgmlType, GgufTensorInfo, GgufValue};
use super::{
    Result, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, MAX_GGUF_HEADER_BYTES,
    MAX_GGUF_TENSOR_RANK, QUANT_BLOCK_LAYOUT,
};

/// Owned tensor payloads with a shared view of supported model encodings.
#[derive(Debug, Clone)]
pub struct PackedTensorReader {
    pub(crate) bytes: Vec<u8>,
    pub(crate) metadata: BTreeMap<String, GgufValue>,
    pub(crate) tensors: Vec<GgufTensorInfo>,
    pub(crate) data_start: usize,
    pub(crate) embedded_manifest: Option<ModelManifest>,
}

pub type GgufReader = PackedTensorReader;

impl PackedTensorReader {
    /// Open GGUF or QIF based on its file magic.
    pub fn open_artifact(bytes: Vec<u8>) -> Result<Self> {
        if qif_v2::has_magic(&bytes) {
            qif_v2::open_packed(bytes)
        } else if qif::has_magic(&bytes) {
            qif::open_packed(bytes)
        } else {
            Self::open(bytes)
        }
    }

    pub fn embedded_manifest(&self) -> Option<&ModelManifest> {
        self.embedded_manifest.as_ref()
    }

    /// Detect GGML dimension order and packed projection rows from generic
    /// GGUF metadata. Architecture-specific interpretation belongs to model
    /// adapters, not the format reader.
    pub fn uses_ggml_output_rows(&self) -> Result<bool> {
        if self.embedded_manifest.is_some() {
            return qif::uses_ggml_dimension_order(&self.metadata);
        }
        if self
            .metadata
            .keys()
            .any(|key| key.ends_with(".tie_word_embeddings"))
        {
            return Ok(false);
        }
        let length = self
            .metadata
            .iter()
            .find(|(key, _)| key.ends_with(".embedding_length"));
        let Ok(embedding) = self.tensor_info("token_embd.weight") else {
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
        Ok(self.metadata.contains_key("general.file_type"))
    }

    pub(crate) fn uses_legacy_quant_block_layout(&self) -> Result<bool> {
        if self.embedded_manifest.is_some() {
            return Ok(false);
        }
        match self.metadata.get(QUANT_BLOCK_LAYOUT) {
            Some(GgufValue::String(layout)) if layout == "ggml" => Ok(false),
            Some(_) => Err(FormatError::Invalid(
                "unsupported GGUF quantization block layout".to_owned(),
            )),
            // Older QuantInfer converter outputs declare the architecture-specific
            // tie flag. The old standalone writer could omit the architecture.
            None => Ok(match self.metadata.get("general.architecture") {
                Some(GgufValue::String(architecture)) => self
                    .metadata
                    .contains_key(&format!("{architecture}.tie_word_embeddings")),
                _ => true,
            }),
        }
    }

    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        let mut cursor = Cursor::new(&bytes);
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
            if data_end > bytes.len() {
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
        Ok(Self {
            bytes,
            metadata,
            tensors,
            data_start,
            embedded_manifest: None,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn file_size_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
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
