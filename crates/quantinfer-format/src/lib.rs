//! Compatibility format readers and writers.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, Read, Seek, Write};

use half::{bf16, f16};
use quantinfer_core::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use serde::Serialize;

pub type Result<T> = std::result::Result<T, FormatError>;

pub mod manifest;
pub mod qif;
pub mod safetensors;

pub use manifest::*;
pub use safetensors::{
    SafeTensorDType, SafeTensorFile, SafeTensorFileReader, SafeTensorInfo, SafeTensorShards,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const QUANT_BLOCK_LAYOUT: &str = "quantinfer.ggml.block_layout";

#[derive(Debug, Clone, PartialEq)]
pub enum FormatError {
    Invalid(String),
    Io(String),
    Core(quantinfer_core::CoreError),
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Io(message) => formatter.write_str(message),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FormatError {}

impl From<quantinfer_core::CoreError> for FormatError {
    fn from(error: quantinfer_core::CoreError) -> Self {
        Self::Core(error)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_0 = 8,
    BF16 = 30,
}

impl TryFrom<u32> for GgmlType {
    type Error = FormatError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            8 => Ok(Self::Q8_0),
            30 => Ok(Self::BF16),
            _ => Err(FormatError::Invalid(format!(
                "unsupported GGML tensor type {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: u64,
    pub size: usize,
}

#[derive(Debug, Clone)]
struct PendingTensor {
    name: String,
    shape: Vec<usize>,
    dtype: GgmlType,
    data: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct GgufWriter {
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<PendingTensor>,
    alignment: u64,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            ..Self::default()
        }
    }

    pub fn insert_metadata(&mut self, key: impl Into<String>, value: GgufValue) {
        self.metadata.insert(key.into(), value);
    }

    pub fn add_f32_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let shape = shape.into();
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "F32 tensor shape does not match data length".to_owned(),
            ));
        }
        let data_size = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| FormatError::Invalid("F32 tensor size overflows usize".to_owned()))?;
        let mut data = Vec::with_capacity(data_size);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape,
            dtype: GgmlType::F32,
            data,
        });
        Ok(())
    }

    pub fn add_f16_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let shape = shape.into();
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "F16 tensor shape does not match data length".to_owned(),
            ));
        }
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<f16>());
        for value in values {
            data.extend_from_slice(&f16::from_f32(*value).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape,
            dtype: GgmlType::F16,
            data,
        });
        Ok(())
    }

    pub fn add_bf16_tensor(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        let shape = shape.into();
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
        })?;
        if elements != values.len() {
            return Err(FormatError::Invalid(
                "BF16 tensor shape does not match data length".to_owned(),
            ));
        }
        let mut data = Vec::with_capacity(values.len() * std::mem::size_of::<bf16>());
        for value in values {
            data.extend_from_slice(&bf16::from_f32(*value).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape,
            dtype: GgmlType::BF16,
            data,
        });
        Ok(())
    }

    pub fn add_q8_0_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensor) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 34);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend(block.values.iter().map(|value| *value as u8));
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q8_0,
            data,
        });
    }

    pub fn add_q4_0_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensor4) {
        self.insert_metadata(QUANT_BLOCK_LAYOUT, GgufValue::String("ggml".to_owned()));
        let mut data = Vec::with_capacity(tensor.blocks().len() * 18);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            let (lower, upper) = block.values.split_at(Q4_0_BLOCK_SIZE / 2);
            data.extend(lower.iter().zip(upper).map(|(low, high)| low | (high << 4)));
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q4_0,
            data,
        });
    }

    pub fn add_q4_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ4K) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 144);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend_from_slice(&f16::from_f32(block.min_scale).to_le_bytes());
            data.extend_from_slice(&block.scales);
            data.extend_from_slice(&block.values);
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q4_K,
            data,
        });
    }

    pub fn add_q5_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ5K) {
        let mut data = Vec::with_capacity(tensor.blocks().len() * 176);
        for block in tensor.blocks() {
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
            data.extend_from_slice(&f16::from_f32(block.min_scale).to_le_bytes());
            data.extend_from_slice(&block.scales);
            data.extend_from_slice(&block.high_bits);
            data.extend_from_slice(&block.values);
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q5_K,
            data,
        });
    }

    pub fn add_q6_k_tensor(&mut self, name: impl Into<String>, tensor: &QuantizedTensorQ6K) {
        self.insert_metadata(QUANT_BLOCK_LAYOUT, GgufValue::String("ggml".to_owned()));
        let mut data = Vec::with_capacity(tensor.blocks().len() * 210);
        for block in tensor.blocks() {
            data.extend_from_slice(&block.low_bits);
            data.extend_from_slice(&block.high_bits);
            data.extend(block.scales.iter().map(|value| *value as u8));
            data.extend_from_slice(&f16::from_f32(block.scale).to_le_bytes());
        }
        self.tensors.push(PendingTensor {
            name: name.into(),
            shape: tensor.shape().dims().to_vec(),
            dtype: GgmlType::Q6_K,
            data,
        });
    }

    pub fn finish(self) -> Result<Vec<u8>> {
        let mut header = GgufHeader::new(self.alignment);
        let padding = self
            .tensors
            .iter()
            .map(|tensor| header.append(tensor))
            .collect::<Result<Vec<_>>>()?;
        let data_size = usize::try_from(header.data_size)
            .map_err(|_| FormatError::Invalid("GGUF data size overflows usize".to_owned()))?;
        let mut output = header.encode(self.metadata)?;
        output.try_reserve_exact(data_size).map_err(|error| {
            FormatError::Invalid(format!("cannot allocate GGUF output: {error}"))
        })?;
        for (tensor, padding) in self.tensors.into_iter().zip(padding) {
            output.resize(output.len() + padding, 0);
            output.extend_from_slice(&tensor.data);
        }
        Ok(output)
    }
}

struct GgufHeader {
    names: std::collections::BTreeSet<String>,
    infos: Vec<u8>,
    alignment: u64,
    data_size: u64,
}

impl GgufHeader {
    fn new(alignment: u64) -> Self {
        Self {
            names: Default::default(),
            infos: Vec::new(),
            alignment,
            data_size: 0,
        }
    }

    fn append(&mut self, tensor: &PendingTensor) -> Result<usize> {
        if !self.names.insert(tensor.name.clone()) {
            return Err(FormatError::Invalid(format!(
                "duplicate GGUF tensor name {}",
                tensor.name
            )));
        }
        let offset = align_up(self.data_size, self.alignment)?;
        let padding = (offset - self.data_size) as usize;
        encode_tensor_info(&mut self.infos, tensor, offset)?;
        self.data_size = offset
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| FormatError::Invalid("GGUF data size overflows u64".to_owned()))?;
        Ok(padding)
    }

    fn encode(self, mut metadata: BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
        metadata
            .entry("general.alignment".to_owned())
            .or_insert(GgufValue::U32(self.alignment as u32));
        let mut output = Vec::new();
        output.extend_from_slice(GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(self.names.len() as u64).to_le_bytes());
        output.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        output.extend_from_slice(&encode_metadata(&metadata)?);
        output.extend_from_slice(&self.infos);
        let data_start = usize::try_from(align_up(output.len() as u64, self.alignment)?)
            .map_err(|_| FormatError::Invalid("GGUF data offset overflows usize".to_owned()))?;
        output.resize(data_start, 0);
        Ok(output)
    }
}

/// Writes tensor payloads to a scratch stream while retaining only the GGUF header.
pub struct GgufSpool<F> {
    scratch: F,
    header: Option<GgufHeader>,
    writer: GgufWriter,
}

impl<F: Read + Seek + Write> GgufSpool<F> {
    /// The scratch stream must be empty and positioned at its beginning.
    pub fn new(scratch: F, writer: GgufWriter) -> Self {
        Self {
            scratch,
            header: Some(GgufHeader::new(writer.alignment)),
            writer,
        }
    }

    /// Encode a tensor with the usual writer and immediately spool its payload.
    pub fn append(&mut self, encode: impl FnOnce(&mut GgufWriter) -> Result<()>) -> Result<()> {
        let mut header = self.header.take().ok_or_else(|| {
            FormatError::Invalid("cannot continue a failed GGUF stream".to_owned())
        })?;
        encode(&mut self.writer)?;
        for tensor in self.writer.tensors.drain(..) {
            let padding = header.append(&tensor)?;
            self.scratch
                .write_all(&[0; DEFAULT_ALIGNMENT as usize][..padding])
                .and_then(|()| self.scratch.write_all(&tensor.data))
                .map_err(|error| FormatError::Io(format!("cannot spool GGUF tensor: {error}")))?;
        }
        self.header = Some(header);
        Ok(())
    }

    pub fn finish(mut self, output: &mut impl Write) -> Result<()> {
        self.append(|_| Ok(()))?;
        let header = self.header.take().expect("append validated the header");
        let data_size = header.data_size;
        let bytes = header.encode(self.writer.metadata)?;
        let copied = self
            .scratch
            .rewind()
            .and_then(|()| output.write_all(&bytes))
            .and_then(|()| io::copy(&mut self.scratch.take(data_size), output))
            .map_err(|error| FormatError::Io(format!("cannot write GGUF: {error}")))?;
        if copied != data_size {
            return Err(FormatError::Io(
                "GGUF scratch stream was truncated".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Owned tensor payloads with a shared view of supported model encodings.
#[derive(Debug, Clone)]
pub struct PackedTensorReader {
    bytes: Vec<u8>,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    data_start: usize,
    embedded_manifest: Option<ModelManifest>,
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

    /// Llama GGML files store output rows and use fastest-axis-first dimensions.
    pub fn llama_uses_ggml_layout(&self) -> Result<bool> {
        self.uses_ggml_output_rows("llama")
    }

    /// Detect GGML dimension order and packed projection rows for a Dense model.
    pub fn uses_ggml_output_rows(&self, architecture: &str) -> Result<bool> {
        if !matches!(architecture, "llama" | "qwen3") {
            return Ok(false);
        }
        if self.embedded_manifest.is_some() {
            return qif::uses_ggml_dimension_order(&self.metadata);
        }
        if self
            .metadata
            .contains_key(&format!("{architecture}.tie_word_embeddings"))
        {
            return Ok(false);
        }
        let length_key = format!("{architecture}.embedding_length");
        if architecture == "qwen3"
            && !self.metadata.contains_key(&length_key)
            && !self.metadata.contains_key("general.file_type")
        {
            return Ok(false);
        }
        let embedding = self.tensor_info("token_embd.weight")?;
        if let (Some(value), [first, second]) =
            (self.metadata.get(&length_key), embedding.shape.as_slice())
        {
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

    fn uses_legacy_quant_block_layout(&self) -> Result<bool> {
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
            metadata.insert(key, cursor.read_value()?);
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
            let mut shape = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                shape.push(usize::try_from(cursor.read_u64()?).map_err(|_| {
                    FormatError::Invalid("GGUF tensor dimension overflows usize".to_owned())
                })?);
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
        let data_start = usize::try_from(align_up(cursor.position() as u64, alignment)?)
            .map_err(|_| FormatError::Invalid("GGUF data offset overflows usize".to_owned()))?;
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

fn encode_tensor_info(output: &mut Vec<u8>, tensor: &PendingTensor, offset: u64) -> Result<()> {
    write_string(output, &tensor.name)?;
    let dimensions = u32::try_from(tensor.shape.len())
        .map_err(|_| FormatError::Invalid("GGUF tensor rank overflows u32".to_owned()))?;
    output.extend_from_slice(&dimensions.to_le_bytes());
    for dimension in &tensor.shape {
        output.extend_from_slice(
            &u64::try_from(*dimension)
                .map_err(|_| FormatError::Invalid("GGUF dimension overflows u64".to_owned()))?
                .to_le_bytes(),
        );
    }
    output.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
    output.extend_from_slice(&offset.to_le_bytes());
    Ok(())
}

fn encode_metadata(metadata: &BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    for (key, value) in metadata {
        write_string(&mut output, key)?;
        encode_value(&mut output, value)?;
    }
    Ok(output)
}

fn encode_value(output: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
    if let GgufValue::Array(values) = value {
        let Some(first) = values.first() else {
            return Err(FormatError::Invalid(
                "GGUF metadata arrays cannot be empty".to_owned(),
            ));
        };
        let element_kind = value_kind(first)?;
        if values
            .iter()
            .any(|item| value_kind(item) != Ok(element_kind))
        {
            return Err(FormatError::Invalid(
                "GGUF metadata arrays must contain one value type".to_owned(),
            ));
        }
        output.extend_from_slice(&9_u32.to_le_bytes());
        output.extend_from_slice(&element_kind.to_le_bytes());
        output.extend_from_slice(
            &u64::try_from(values.len())
                .map_err(|_| FormatError::Invalid("GGUF metadata array is too large".to_owned()))?
                .to_le_bytes(),
        );
        for item in values {
            encode_scalar(output, item)?;
        }
        return Ok(());
    }
    output.extend_from_slice(&value_kind(value)?.to_le_bytes());
    encode_scalar(output, value)
}

fn value_kind(value: &GgufValue) -> Result<u32> {
    match value {
        GgufValue::U32(_) => Ok(4),
        GgufValue::U64(_) => Ok(10),
        GgufValue::I32(_) => Ok(5),
        GgufValue::I64(_) => Ok(11),
        GgufValue::F32(_) => Ok(6),
        GgufValue::F64(_) => Ok(12),
        GgufValue::Bool(_) => Ok(7),
        GgufValue::String(_) => Ok(8),
        GgufValue::Array(_) => Err(FormatError::Invalid(
            "nested GGUF metadata arrays are not supported".to_owned(),
        )),
    }
}

fn encode_scalar(output: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::U32(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::U64(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::I32(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::I64(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::F32(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::F64(value) => output.extend_from_slice(&value.to_le_bytes()),
        GgufValue::Bool(value) => output.push(u8::from(*value)),
        GgufValue::String(value) => write_string(output, value)?,
        GgufValue::Array(_) => {
            return Err(FormatError::Invalid(
                "nested GGUF metadata arrays are not supported".to_owned(),
            ));
        }
    }
    Ok(())
}

fn tensor_byte_size(dtype: GgmlType, shape: &[usize]) -> Result<usize> {
    let elements = shape.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| FormatError::Invalid("tensor shape overflows usize".to_owned()))
    })?;
    match dtype {
        GgmlType::F32 => elements
            .checked_mul(4)
            .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned())),
        GgmlType::F16 | GgmlType::BF16 => elements
            .checked_mul(2)
            .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned())),
        GgmlType::Q4_0 => {
            if !elements.is_multiple_of(Q4_0_BLOCK_SIZE) {
                return Err(FormatError::Invalid(
                    "Q4_0 tensor is not block aligned".to_owned(),
                ));
            }
            (elements / Q4_0_BLOCK_SIZE)
                .checked_mul(2 + Q4_0_BLOCK_SIZE / 2)
                .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned()))
        }
        GgmlType::Q4_K => {
            if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) {
                return Err(FormatError::Invalid(
                    "Q4_K tensor is not block aligned".to_owned(),
                ));
            }
            (elements / Q4_K_BLOCK_SIZE)
                .checked_mul(144)
                .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned()))
        }
        GgmlType::Q5_K => {
            if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) {
                return Err(FormatError::Invalid(
                    "Q5_K tensor is not block aligned".to_owned(),
                ));
            }
            (elements / Q4_K_BLOCK_SIZE)
                .checked_mul(176)
                .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned()))
        }
        GgmlType::Q6_K => {
            if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) {
                return Err(FormatError::Invalid(
                    "Q6_K tensor is not block aligned".to_owned(),
                ));
            }
            (elements / Q4_K_BLOCK_SIZE)
                .checked_mul(210)
                .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned()))
        }
        GgmlType::Q8_0 => {
            if !elements.is_multiple_of(32) {
                return Err(FormatError::Invalid(
                    "Q8_0 tensor is not block aligned".to_owned(),
                ));
            }
            (elements / 32)
                .checked_mul(34)
                .ok_or_else(|| FormatError::Invalid("tensor size overflows usize".to_owned()))
        }
    }
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output.extend_from_slice(
        &u64::try_from(value.len())
            .map_err(|_| FormatError::Invalid("GGUF string is too large".to_owned()))?
            .to_le_bytes(),
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(FormatError::Invalid(
            "GGUF alignment must be non-zero".to_owned(),
        ));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .ok_or_else(|| FormatError::Invalid("GGUF alignment overflows u64".to_owned()))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn read_exact(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| FormatError::Invalid("GGUF cursor overflows usize".to_owned()))?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| FormatError::Invalid("GGUF data is truncated".to_owned()))?;
        self.position = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| FormatError::Invalid("GGUF string length overflows usize".to_owned()))?;
        String::from_utf8(self.read_exact(length)?.to_vec())
            .map_err(|_| FormatError::Invalid("GGUF string is not UTF-8".to_owned()))
    }

    fn read_value(&mut self) -> Result<GgufValue> {
        self.read_value_with_limit(usize::MAX)
    }

    fn read_value_with_limit(&mut self, array_limit: usize) -> Result<GgufValue> {
        let kind = self.read_u32()?;
        if kind == 9 {
            let element_kind = self.read_u32()?;
            let length = usize::try_from(self.read_u64()?).map_err(|_| {
                FormatError::Invalid("GGUF metadata array length overflows usize".to_owned())
            })?;
            if length > array_limit || length > self.bytes.len().saturating_sub(self.position) {
                return Err(FormatError::Invalid(
                    "GGUF metadata array exceeds file capacity".to_owned(),
                ));
            }
            let mut values = Vec::with_capacity(length);
            for _ in 0..length {
                values.push(self.read_scalar(element_kind)?);
            }
            return Ok(GgufValue::Array(values));
        }
        self.read_scalar(kind)
    }

    fn read_scalar(&mut self, kind: u32) -> Result<GgufValue> {
        match kind {
            0 => Ok(GgufValue::U32(u32::from(self.read_exact(1)?[0]))),
            1 => Ok(GgufValue::I32(i32::from(i8::from_le_bytes([
                self.read_exact(1)?[0]
            ])))),
            2 => Ok(GgufValue::U32(u32::from(u16::from_le_bytes(
                self.read_exact(2)?.try_into().unwrap(),
            )))),
            3 => Ok(GgufValue::I32(i32::from(i16::from_le_bytes(
                self.read_exact(2)?.try_into().unwrap(),
            )))),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            5 => Ok(GgufValue::I32(self.read_u32()? as i32)),
            11 => Ok(GgufValue::I64(self.read_u64()? as i64)),
            6 => Ok(GgufValue::F32(f32::from_le_bytes(
                self.read_exact(4)?
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid F32 metadata value".to_owned()))?,
            ))),
            7 => Ok(GgufValue::Bool(self.read_exact(1)?[0] != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            12 => Ok(GgufValue::F64(f64::from_le_bytes(
                self.read_exact(8)?.try_into().unwrap(),
            ))),
            kind => Err(FormatError::Invalid(format!(
                "unsupported GGUF metadata type {kind}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FormatError, GgmlType, GgufReader, GgufSpool, GgufValue, GgufWriter, QUANT_BLOCK_LAYOUT,
    };
    use quantinfer_core::{
        Q4_0Block, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
        QuantizedTensorQ6K, Tensor,
    };

    #[test]
    fn round_trips_scalar_metadata_and_f32_tensor() {
        let mut writer = GgufWriter::new();
        writer.insert_metadata(
            "general.architecture",
            GgufValue::String("qwen3".to_owned()),
        );
        writer.insert_metadata("qwen3.context_length", GgufValue::U32(128));
        writer
            .add_f32_tensor("token_embd.weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        writer
            .add_f16_tensor("f16.weight", vec![2], &[1.0, 2.0])
            .unwrap();
        writer
            .add_bf16_tensor("bf16.weight", vec![2], &[3.0, 4.0])
            .unwrap();
        let bytes = writer.finish().unwrap();
        let reader = GgufReader::open(bytes).unwrap();
        assert_eq!(
            reader.metadata().get("general.architecture"),
            Some(&GgufValue::String("qwen3".to_owned()))
        );
        assert_eq!(reader.tensors()[0].shape, vec![2, 2]);
        assert_eq!(reader.tensor_bytes("token_embd.weight").unwrap().len(), 16);
        assert_eq!(
            reader.tensor_f32("token_embd.weight").unwrap(),
            Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        );
        assert_eq!(
            reader.tensor_f32("f16.weight").unwrap(),
            Tensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap()
        );
        assert_eq!(
            reader.tensor_f32("bf16.weight").unwrap(),
            Tensor::from_f32(vec![2], vec![3.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn round_trips_metadata_arrays_and_q8_0_tensor() {
        let mut writer = GgufWriter::new();
        writer.insert_metadata(
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![
                GgufValue::String("<|endoftext|>".to_owned()),
                GgufValue::String("hello".to_owned()),
            ]),
        );
        let quantized = QuantizedTensor::from_blocks(
            vec![32],
            vec![Q8_0Block {
                scale: 0.25,
                values: [1_i8; 32],
            }],
        )
        .unwrap();
        writer.add_q8_0_tensor("blk.0.attn_q.weight", &quantized);
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        assert_eq!(
            reader.metadata().get("tokenizer.ggml.tokens"),
            Some(&GgufValue::Array(vec![
                GgufValue::String("<|endoftext|>".to_owned()),
                GgufValue::String("hello".to_owned()),
            ]))
        );
        assert_eq!(
            reader.tensor_info("blk.0.attn_q.weight").unwrap().dtype,
            GgmlType::Q8_0
        );
        assert_eq!(
            reader.tensor_q8_0("blk.0.attn_q.weight").unwrap(),
            quantized
        );
    }

    #[test]
    fn round_trips_q4_0_tensor_with_ggml_packing() {
        let mut values = [0_u8; 32];
        for (index, value) in values.iter_mut().enumerate() {
            *value = index.min(31 - index) as u8;
        }
        let quantized =
            QuantizedTensor4::from_blocks(vec![32], vec![Q4_0Block { scale: 0.5, values }])
                .unwrap();
        let mut writer = GgufWriter::new();
        writer.add_q4_0_tensor("blk.0.attn_q.weight", &quantized);
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        assert_eq!(
            reader.tensor_info("blk.0.attn_q.weight").unwrap().dtype,
            GgmlType::Q4_0
        );
        assert_eq!(
            reader.tensor_bytes("blk.0.attn_q.weight").unwrap(),
            [
                0x00, 0x38, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b,
                0x3c, 0x2d, 0x1e, 0x0f
            ],
        );
        assert_eq!(
            reader.tensor_q4_0("blk.0.attn_q.weight").unwrap(),
            quantized
        );
    }

    #[test]
    fn spools_tensors_with_identical_metadata_and_alignment() {
        let quantized = QuantizedTensor4::from_blocks(
            vec![32],
            vec![Q4_0Block {
                scale: 0.5,
                values: [8; 32],
            }],
        )
        .unwrap();
        let mut writer = GgufWriter::new();
        writer.insert_metadata("general.architecture", GgufValue::String("llama".into()));
        let mut spool = GgufSpool::new(std::io::Cursor::new(Vec::new()), writer.clone());
        let add_f32 = |writer: &mut GgufWriter| writer.add_f32_tensor("norm", vec![1], &[2.0]);
        let add_q4 = |writer: &mut GgufWriter| {
            writer.add_q4_0_tensor("projection", &quantized);
            Ok(())
        };
        add_f32(&mut writer).unwrap();
        add_q4(&mut writer).unwrap();
        spool.append(add_f32).unwrap();
        spool.append(add_q4).unwrap();
        let mut output = Vec::new();
        spool.finish(&mut output).unwrap();
        assert_eq!(output, writer.finish().unwrap());
        let reader = GgufReader::open(output).unwrap();
        assert_eq!(reader.tensor_info("projection").unwrap().offset, 32);
        assert_eq!(reader.tensor_q4_0("projection").unwrap(), quantized);
    }

    #[test]
    fn reads_standard_and_legacy_quant_block_bytes() {
        let q4_values = std::array::from_fn(|index| (index % 16) as u8);
        let q4 = QuantizedTensor4::from_blocks(
            vec![32],
            vec![Q4_0Block {
                scale: 0.5,
                values: q4_values,
            }],
        )
        .unwrap();
        let q6 = QuantizedTensorQ6K::from_blocks(
            vec![256],
            vec![Q6KBlock {
                scale: 0.5,
                scales: [-2; 16],
                low_bits: [0x12; 128],
                high_bits: [0x34; 64],
            }],
        )
        .unwrap();
        let mut writer = GgufWriter::new();
        writer.insert_metadata(
            "general.architecture",
            GgufValue::String("llama".to_owned()),
        );
        writer.add_q4_0_tensor("q4", &q4);
        writer.add_q6_k_tensor("q6", &q6);
        let standard_q6 = [&[0x12; 128][..], &[0x34; 64], &[0xfe; 16], &[0x00, 0x38]].concat();
        writer.metadata.remove(QUANT_BLOCK_LAYOUT);
        let reader = GgufReader::open(writer.clone().finish().unwrap()).unwrap();
        assert_eq!(reader.tensor_bytes("q6").unwrap(), standard_q6);
        assert_eq!(reader.tensor_q4_0("q4").unwrap(), q4);
        assert_eq!(reader.tensor_q6_k("q6").unwrap(), q6);

        writer.insert_metadata("llama.tie_word_embeddings", GgufValue::Bool(true));
        writer.tensors[0].data = vec![
            0x00, 0x38, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76,
            0x98, 0xba, 0xdc, 0xfe,
        ];
        writer.tensors[1].data = [
            &standard_q6[208..],
            &standard_q6[192..208],
            &standard_q6[..192],
        ]
        .concat();
        let legacy = GgufReader::open(writer.finish().unwrap()).unwrap();
        assert_eq!(legacy.tensor_q4_0("q4").unwrap(), q4);
        assert_eq!(legacy.tensor_q6_k("q6").unwrap(), q6);
    }

    #[test]
    fn round_trips_q4_k_tensor_with_standard_ggml_layout() {
        let quantized = QuantizedTensorQ4K::from_blocks(
            vec![256],
            vec![quantinfer_core::Q4KBlock {
                scale: 0.125,
                min_scale: 0.25,
                scales: [63; 12],
                values: [0x21; 128],
            }],
        )
        .unwrap();
        let mut writer = GgufWriter::new();
        writer.add_q4_k_tensor("blk.0.attn_q.weight", &quantized);
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let restored: QuantizedTensorQ4K = reader.tensor_q4_k("blk.0.attn_q.weight").unwrap();
        assert_eq!(restored, quantized);
        assert_eq!(reader.tensor_info("blk.0.attn_q.weight").unwrap().size, 144);
    }

    #[test]
    fn rejects_zero_alignment_in_untrusted_gguf() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&17_u64.to_le_bytes());
        bytes.extend_from_slice(b"general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(
            GgufReader::open(bytes),
            Err(FormatError::Invalid(_))
        ));
    }

    #[test]
    fn reads_all_standard_scalar_metadata_kinds() {
        // The reader-level coverage for the compact scalar kinds is exercised
        // through a hand-built metadata section because GgufWriter emits the
        // canonical wide kinds only.
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3_u32.to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
        raw.extend_from_slice(&5_u64.to_le_bytes());
        for (name, kind, value) in [
            ("u8", 0_u32, vec![7]),
            ("i8", 1_u32, vec![0xfb]),
            ("u16", 2_u32, 0x1234_u16.to_le_bytes().to_vec()),
            ("i16", 3_u32, (-7_i16).to_le_bytes().to_vec()),
            ("f64", 12_u32, 1.25_f64.to_le_bytes().to_vec()),
        ] {
            raw.extend_from_slice(&(name.len() as u64).to_le_bytes());
            raw.extend_from_slice(name.as_bytes());
            raw.extend_from_slice(&kind.to_le_bytes());
            raw.extend_from_slice(&value);
        }
        let reader = GgufReader::open(raw).unwrap();
        assert_eq!(reader.metadata().get("f64"), Some(&GgufValue::F64(1.25)));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3_u32.to_le_bytes());
        raw.extend_from_slice(&2_u64.to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
        for _ in 0..2 {
            raw.extend_from_slice(&3_u64.to_le_bytes());
            raw.extend_from_slice(b"dup");
            raw.extend_from_slice(&0_u32.to_le_bytes());
            raw.extend_from_slice(&0_u32.to_le_bytes());
            raw.extend_from_slice(&0_u64.to_le_bytes());
        }
        assert!(matches!(
            GgufReader::open(raw),
            Err(FormatError::Invalid(message)) if message.contains("duplicate GGUF tensor name")
        ));
    }
}
