//! GGUF binary encoding, size checks, and bounded cursor.

use std::collections::BTreeMap;

use quantinfer_core::{Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE};

use super::types::{FormatError, GgmlType, GgufValue, PendingTensor};
use super::{Result, MAX_GGUF_ARRAY_ITEMS};

pub(crate) fn encode_tensor_info(
    output: &mut Vec<u8>,
    tensor: &PendingTensor,
    offset: u64,
) -> Result<()> {
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

pub(crate) fn encode_metadata(metadata: &BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    for (key, value) in metadata {
        write_string(&mut output, key)?;
        encode_value(&mut output, value)?;
    }
    Ok(output)
}

pub(crate) fn encode_value(output: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
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

pub(crate) fn tensor_byte_size(dtype: GgmlType, shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Err(FormatError::Invalid(
            "tensor dimensions must be greater than zero".to_owned(),
        ));
    }
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

pub(crate) fn write_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    output.extend_from_slice(
        &u64::try_from(value.len())
            .map_err(|_| FormatError::Invalid("GGUF string is too large".to_owned()))?
            .to_le_bytes(),
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Result<u64> {
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

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn read_exact(&mut self, length: usize) -> Result<&'a [u8]> {
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

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    pub(crate) fn read_string(&mut self) -> Result<String> {
        let length = usize::try_from(self.read_u64()?)
            .map_err(|_| FormatError::Invalid("GGUF string length overflows usize".to_owned()))?;
        String::from_utf8(self.read_exact(length)?.to_vec())
            .map_err(|_| FormatError::Invalid("GGUF string is not UTF-8".to_owned()))
    }

    pub(crate) fn read_value(&mut self) -> Result<GgufValue> {
        self.read_value_with_limit(MAX_GGUF_ARRAY_ITEMS)
    }

    pub(crate) fn read_value_with_limit(&mut self, array_limit: usize) -> Result<GgufValue> {
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
