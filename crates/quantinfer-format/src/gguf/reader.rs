//! GGUF and QIF reader plus tensor decoding.

use std::collections::{BTreeMap, BTreeSet};

use half::{bf16, f16};
use quantinfer_core::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};

use crate::{manifest::ModelManifest, qif};

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
        if qif::has_magic(&bytes) {
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

    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let info = self.tensor_info(name)?;
        let bytes = self.tensor_bytes(name)?;
        let values: Vec<f32> = match info.dtype {
            GgmlType::F32 => {
                let (chunks, remainder) = bytes.as_chunks::<4>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF F32 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect()
            }
            GgmlType::F16 => {
                let (chunks, remainder) = bytes.as_chunks::<2>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF F16 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
                    .collect()
            }
            GgmlType::BF16 => {
                let (chunks, remainder) = bytes.as_chunks::<2>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF BF16 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                    .collect()
            }
            GgmlType::Q8_0 => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q8_0, expected floating point"
                )))
            }
            GgmlType::Q4_0 => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q4_0, expected floating point"
                )))
            }
            GgmlType::Q4_K => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q4_K, expected floating point"
                )))
            }
            GgmlType::Q5_K | GgmlType::Q6_K => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has a K-quant type, expected floating point"
                )))
            }
        };
        if values.len() != info.shape.iter().product::<usize>() {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has an invalid byte length"
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} contains non-finite values"
            )));
        }
        Ok(Tensor::from_f32(info.shape.clone(), values)?)
    }

    pub fn tensor_q8_0(&self, name: &str) -> Result<QuantizedTensor> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q8_0 {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q8_0",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let block_size = std::mem::size_of::<f16>() + Q8_0_BLOCK_SIZE;
        if !bytes.len().is_multiple_of(block_size) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q8_0 tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / block_size);
        for chunk in bytes.chunks_exact(block_size) {
            let scale = f16::from_le_bytes(
                chunk[..std::mem::size_of::<f16>()]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q8_0 scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q8_0 tensor {name} contains a non-finite scale"
                )));
            }
            let mut values = [0_i8; Q8_0_BLOCK_SIZE];
            for (index, value) in chunk[std::mem::size_of::<f16>()..]
                .iter()
                .copied()
                .enumerate()
            {
                values[index] = i8::from_le_bytes([value]);
            }
            blocks.push(Q8_0Block { scale, values });
        }
        Ok(QuantizedTensor::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q4_0(&self, name: &str) -> Result<QuantizedTensor4> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q4_0 {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q4_0",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let legacy = self.uses_legacy_quant_block_layout()?;
        let block_size = std::mem::size_of::<f16>() + Q4_0_BLOCK_SIZE / 2;
        if !bytes.len().is_multiple_of(block_size) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q4_0 tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / block_size);
        for chunk in bytes.chunks_exact(block_size) {
            let scale = f16::from_le_bytes(
                chunk[..std::mem::size_of::<f16>()]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_0 scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q4_0 tensor {name} contains a non-finite scale"
                )));
            }
            let mut values = [0_u8; Q4_0_BLOCK_SIZE];
            for (index, packed) in chunk[std::mem::size_of::<f16>()..]
                .iter()
                .copied()
                .enumerate()
            {
                let (low, high) = if legacy {
                    (index * 2, index * 2 + 1)
                } else {
                    (index, index + Q4_0_BLOCK_SIZE / 2)
                };
                values[low] = packed & 0x0f;
                values[high] = packed >> 4;
            }
            blocks.push(Q4_0Block { scale, values });
        }
        Ok(QuantizedTensor4::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q4_k(&self, name: &str) -> Result<QuantizedTensorQ4K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q4_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q4_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if !bytes.len().is_multiple_of(144) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q4_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 144);
        for chunk in bytes.as_chunks::<144>().0 {
            let scale = f16::from_le_bytes(
                chunk[..2]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_K scale".to_owned()))?,
            )
            .to_f32();
            let min_scale = f16::from_le_bytes(
                chunk[2..4]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_K minimum scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() || !min_scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q4_K tensor {name} contains a non-finite scale"
                )));
            }
            let mut scales = [0_u8; 12];
            scales.copy_from_slice(&chunk[4..16]);
            let mut values = [0_u8; Q4_K_BLOCK_SIZE / 2];
            values.copy_from_slice(&chunk[16..]);
            blocks.push(Q4KBlock {
                scale,
                min_scale,
                scales,
                values,
            });
        }
        Ok(QuantizedTensorQ4K::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q5_k(&self, name: &str) -> Result<QuantizedTensorQ5K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q5_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q5_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if !bytes.len().is_multiple_of(176) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q5_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 176);
        for chunk in bytes.as_chunks::<176>().0 {
            let scale = f16::from_le_bytes(chunk[..2].try_into().unwrap()).to_f32();
            let min_scale = f16::from_le_bytes(chunk[2..4].try_into().unwrap()).to_f32();
            if !scale.is_finite() || !min_scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q5_K tensor {name} contains a non-finite scale"
                )));
            }
            let mut scales = [0_u8; 12];
            scales.copy_from_slice(&chunk[4..16]);
            let mut high_bits = [0_u8; 32];
            high_bits.copy_from_slice(&chunk[16..48]);
            let mut values = [0_u8; 128];
            values.copy_from_slice(&chunk[48..]);
            blocks.push(Q5KBlock {
                scale,
                min_scale,
                scales,
                high_bits,
                values,
            });
        }
        Ok(QuantizedTensorQ5K::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q6_k(&self, name: &str) -> Result<QuantizedTensorQ6K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q6_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q6_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let legacy = self.uses_legacy_quant_block_layout()?;
        if !bytes.len().is_multiple_of(210) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q6_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 210);
        for chunk in bytes.as_chunks::<210>().0 {
            let (scale, scales, low_bits, high_bits) = if legacy {
                (&chunk[..2], &chunk[2..18], &chunk[18..146], &chunk[146..])
            } else {
                (
                    &chunk[208..],
                    &chunk[192..208],
                    &chunk[..128],
                    &chunk[128..192],
                )
            };
            let scale = f16::from_le_bytes(scale.try_into().unwrap()).to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q6_K tensor {name} contains a non-finite scale"
                )));
            }
            blocks.push(Q6KBlock {
                scale,
                scales: std::array::from_fn(|index| scales[index] as i8),
                low_bits: low_bits.try_into().unwrap(),
                high_bits: high_bits.try_into().unwrap(),
            });
        }
        Ok(QuantizedTensorQ6K::from_blocks(info.shape.clone(), blocks)?)
    }
}
