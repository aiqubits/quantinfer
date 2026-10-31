//! Compatibility format readers and writers.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use half::f16;
use quantinfer_core::QuantizedTensor;

pub type Result<T> = std::result::Result<T, FormatError>;

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
        if shape.iter().product::<usize>() != values.len() {
            return Err(FormatError::Invalid(
                "F32 tensor shape does not match data length".to_owned(),
            ));
        }
        let mut data = Vec::with_capacity(values.len() * 4);
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
        self.metadata
            .entry("general.alignment".to_owned())
            .or_insert(GgufValue::U32(self.alignment as u32));
        let metadata = encode_metadata(&self.metadata)?;
        let mut infos = Vec::new();
        let tensor_info_size = self.tensors.iter().try_fold(0usize, |size, tensor| {
            Ok::<_, FormatError>(size + tensor_info_size(tensor)?)
        })?;
        let header_size = 4 + 4 + 8 + 8 + metadata.len() + tensor_info_size;
        let data_start = align_up(header_size as u64, self.alignment) as usize;
        let mut data_offset = 0_u64;
        for tensor in &self.tensors {
            data_offset = align_up(data_offset, self.alignment);
            encode_tensor_info(&mut infos, tensor, data_offset)?;
            data_offset += tensor.data.len() as u64;
        }

        let mut output = Vec::with_capacity(data_start + data_offset as usize);
        output.extend_from_slice(GGUF_MAGIC);
        output.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        output.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        output.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        output.extend_from_slice(&metadata);
        output.extend_from_slice(&infos);
        output.resize(data_start, 0);
        for tensor in self.tensors {
            let target = data_start
                + align_up(output.len() as u64 - data_start as u64, self.alignment) as usize;
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
        let tensor_count = cursor.read_u64()? as usize;
        let metadata_count = cursor.read_u64()? as usize;
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
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let dimensions = cursor.read_u32()? as usize;
            let mut shape = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                shape.push(cursor.read_u64()? as usize);
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
        let data_start = align_up(cursor.position() as u64, alignment) as usize;
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
        let info = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown GGUF tensor {name}")))?;
        let start = self.data_start + info.offset as usize;
        let end = start + info.size;
        self.bytes
            .get(start..end)
            .ok_or_else(|| FormatError::Invalid("GGUF tensor data is truncated".to_owned()))
    }
}

fn tensor_info_size(tensor: &PendingTensor) -> Result<usize> {
    Ok(8 + tensor.name.len() + 4 + tensor.shape.len() * 8 + 4 + 8)
}

fn encode_tensor_info(output: &mut Vec<u8>, tensor: &PendingTensor, offset: u64) -> Result<()> {
    write_string(output, &tensor.name)?;
    output.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
    for dimension in &tensor.shape {
        output.extend_from_slice(&(*dimension as u64).to_le_bytes());
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
    let (kind, encoded): (u32, Vec<u8>) = match value {
        GgufValue::U32(value) => (4, value.to_le_bytes().to_vec()),
        GgufValue::U64(value) => (10, value.to_le_bytes().to_vec()),
        GgufValue::I32(value) => (5, value.to_le_bytes().to_vec()),
        GgufValue::I64(value) => (11, value.to_le_bytes().to_vec()),
        GgufValue::F32(value) => (6, value.to_le_bytes().to_vec()),
        GgufValue::Bool(value) => (7, vec![u8::from(*value)]),
        GgufValue::String(value) => {
            let mut bytes = Vec::new();
            write_string(&mut bytes, value)?;
            (8, bytes)
        }
    };
    output.extend_from_slice(&kind.to_le_bytes());
    output.extend_from_slice(&encoded);
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
            if elements % 32 != 0 {
                return Err(FormatError::Invalid(
                    "Q8_0 tensor is not block aligned".to_owned(),
                ));
            }
            Ok(elements / 32 * 34)
        }
    }
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) / alignment * alignment
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
        let end = self.position + length;
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
        let length = self.read_u64()? as usize;
        String::from_utf8(self.read_exact(length)?.to_vec())
            .map_err(|_| FormatError::Invalid("GGUF string is not UTF-8".to_owned()))
    }

    fn read_value(&mut self) -> Result<GgufValue> {
        match self.read_u32()? {
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            5 => Ok(GgufValue::I32(self.read_u32()? as i32)),
            11 => Ok(GgufValue::I64(self.read_u64()? as i64)),
            6 => Ok(GgufValue::F32(f32::from_le_bytes(
                self.read_exact(4)?.try_into().unwrap(),
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
    use super::{GgufReader, GgufValue, GgufWriter};

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
    }
}
