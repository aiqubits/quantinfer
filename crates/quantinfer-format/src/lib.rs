//! Compatibility format readers and writers.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use half::f16;
use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor, Q8_0_BLOCK_SIZE};

pub type Result<T> = std::result::Result<T, FormatError>;

pub mod safetensors;

pub use safetensors::{SafeTensorDType, SafeTensorFile, SafeTensorInfo};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

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

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    Q8_0 = 8,
}

impl TryFrom<u32> for GgmlType {
    type Error = FormatError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            8 => Ok(Self::Q8_0),
            _ => Err(FormatError::Invalid(format!(
                "unsupported GGML tensor type {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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

    pub fn finish(mut self) -> Result<Vec<u8>> {
        let mut names = std::collections::BTreeSet::new();
        for tensor in &self.tensors {
            if !names.insert(&tensor.name) {
                return Err(FormatError::Invalid(format!(
                    "duplicate GGUF tensor name {}",
                    tensor.name
                )));
            }
        }
        self.metadata
            .entry("general.alignment".to_owned())
            .or_insert(GgufValue::U32(self.alignment as u32));
        let metadata = encode_metadata(&self.metadata)?;
        let mut infos = Vec::new();
        let tensor_info_size = self.tensors.iter().try_fold(0usize, |size, tensor| {
            size.checked_add(tensor_info_size(tensor)?)
                .ok_or_else(|| FormatError::Invalid("GGUF header size overflows usize".to_owned()))
        })?;
        let header_size = 4usize
            .checked_add(4)
            .and_then(|size| size.checked_add(8))
            .and_then(|size| size.checked_add(8))
            .and_then(|size| size.checked_add(metadata.len()))
            .and_then(|size| size.checked_add(tensor_info_size))
            .ok_or_else(|| FormatError::Invalid("GGUF header size overflows usize".to_owned()))?;
        let data_start = usize::try_from(align_up(header_size as u64, self.alignment)?)
            .map_err(|_| FormatError::Invalid("GGUF data offset overflows usize".to_owned()))?;
        let mut data_offset = 0_u64;
        for tensor in &self.tensors {
            data_offset = align_up(data_offset, self.alignment)?;
            encode_tensor_info(&mut infos, tensor, data_offset)?;
            data_offset = data_offset
                .checked_add(u64::try_from(tensor.data.len()).map_err(|_| {
                    FormatError::Invalid("GGUF tensor size overflows u64".to_owned())
                })?)
                .ok_or_else(|| FormatError::Invalid("GGUF data size overflows u64".to_owned()))?;
        }

        let data_size = usize::try_from(data_offset)
            .map_err(|_| FormatError::Invalid("GGUF data size overflows usize".to_owned()))?;
        let capacity = data_start
            .checked_add(data_size)
            .ok_or_else(|| FormatError::Invalid("GGUF output size overflows usize".to_owned()))?;
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        output.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        output.extend_from_slice(&metadata);
        output.extend_from_slice(&infos);
        output.resize(data_start, 0);
        for tensor in self.tensors {
            let relative = u64::try_from(output.len() - data_start)
                .map_err(|_| FormatError::Invalid("GGUF output offset overflows u64".to_owned()))?;
            let target = data_start
                .checked_add(
                    usize::try_from(align_up(relative, self.alignment)?).map_err(|_| {
                        FormatError::Invalid("GGUF output offset overflows usize".to_owned())
                    })?,
                )
                .ok_or_else(|| {
                    FormatError::Invalid("GGUF output size overflows usize".to_owned())
                })?;
            output.resize(target, 0);
            output.extend_from_slice(&tensor.data);
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct GgufReader {
    bytes: Vec<u8>,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
    data_start: usize,
}

impl GgufReader {
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
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
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
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
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
        if info.dtype != GgmlType::F32 {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected F32",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let mut values = Vec::with_capacity(info.shape.iter().product::<usize>());
        let (chunks, remainder) = bytes.as_chunks::<4>();
        for chunk in chunks {
            values.push(f32::from_le_bytes(*chunk));
        }
        if !remainder.is_empty() || values.len() != info.shape.iter().product::<usize>() {
            return Err(FormatError::Invalid(format!(
                "GGUF F32 tensor {name} has an invalid byte length"
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
}

fn tensor_info_size(tensor: &PendingTensor) -> Result<usize> {
    8usize
        .checked_add(tensor.name.len())
        .and_then(|size| size.checked_add(4))
        .and_then(|size| size.checked_add(tensor.shape.len().checked_mul(8)?))
        .and_then(|size| size.checked_add(4))
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| FormatError::Invalid("GGUF tensor info size overflows usize".to_owned()))
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
        let kind = self.read_u32()?;
        if kind == 9 {
            let element_kind = self.read_u32()?;
            let length = usize::try_from(self.read_u64()?).map_err(|_| {
                FormatError::Invalid("GGUF metadata array length overflows usize".to_owned())
            })?;
            if length > self.bytes.len().saturating_sub(self.position) {
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
            kind => Err(FormatError::Invalid(format!(
                "unsupported GGUF metadata type {kind}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FormatError, GgmlType, GgufReader, GgufValue, GgufWriter};
    use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor};

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
}
