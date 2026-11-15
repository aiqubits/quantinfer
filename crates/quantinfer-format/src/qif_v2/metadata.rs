use std::collections::BTreeMap;

use crate::{GgufValue, Result};

use super::{invalid, MAX_HEADER_SECTION};

const TAG_U32: u32 = 4;
const TAG_I32: u32 = 5;
const TAG_F32: u32 = 6;
const TAG_BOOL: u32 = 7;
const TAG_STRING: u32 = 8;
const TAG_ARRAY: u32 = 9;
const TAG_U64: u32 = 10;
const TAG_I64: u32 = 11;
const TAG_F64: u32 = 12;
const TAG_MAP: u32 = 13;
const MAX_DEPTH: usize = 32;
const MAX_ITEMS: usize = 1_048_576;
const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum QifV2MetadataValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
    Map(BTreeMap<String, Self>),
}

impl From<&GgufValue> for QifV2MetadataValue {
    fn from(value: &GgufValue) -> Self {
        match value {
            GgufValue::U32(value) => Self::U32(*value),
            GgufValue::U64(value) => Self::U64(*value),
            GgufValue::I32(value) => Self::I32(*value),
            GgufValue::I64(value) => Self::I64(*value),
            GgufValue::F32(value) => Self::F32(*value),
            GgufValue::F64(value) => Self::F64(*value),
            GgufValue::Bool(value) => Self::Bool(*value),
            GgufValue::String(value) => Self::String(value.clone()),
            GgufValue::Array(values) => Self::Array(values.iter().map(Self::from).collect()),
        }
    }
}

impl QifV2MetadataValue {
    pub fn to_gguf(&self) -> Result<GgufValue> {
        match self {
            Self::U32(value) => Ok(GgufValue::U32(*value)),
            Self::U64(value) => Ok(GgufValue::U64(*value)),
            Self::I32(value) => Ok(GgufValue::I32(*value)),
            Self::I64(value) => Ok(GgufValue::I64(*value)),
            Self::F32(value) => Ok(GgufValue::F32(*value)),
            Self::F64(value) => Ok(GgufValue::F64(*value)),
            Self::Bool(value) => Ok(GgufValue::Bool(*value)),
            Self::String(value) => Ok(GgufValue::String(value.clone())),
            Self::Array(values) => values
                .iter()
                .map(Self::to_gguf)
                .collect::<Result<Vec<_>>>()
                .map(GgufValue::Array),
            Self::Map(_) => Err(invalid(
                "metadata maps cannot be projected to GGUF metadata",
            )),
        }
    }
}

pub(crate) fn encode(values: &BTreeMap<String, QifV2MetadataValue>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    write_map(&mut output, values, 0)?;
    if output.len() > MAX_HEADER_SECTION {
        return Err(invalid("metadata exceeds the size limit"));
    }
    Ok(output)
}

pub(crate) fn decode(bytes: &[u8]) -> Result<BTreeMap<String, QifV2MetadataValue>> {
    if bytes.len() > MAX_HEADER_SECTION {
        return Err(invalid("metadata exceeds the size limit"));
    }
    let mut cursor = Cursor { bytes, offset: 0 };
    let values = read_map(&mut cursor, 0)?;
    if cursor.offset != bytes.len() {
        return Err(invalid("unexpected bytes after metadata"));
    }
    Ok(values)
}

fn write_map(
    output: &mut Vec<u8>,
    values: &BTreeMap<String, QifV2MetadataValue>,
    depth: usize,
) -> Result<()> {
    if depth > MAX_DEPTH || values.len() > MAX_ITEMS {
        return Err(invalid("metadata nesting or item limit exceeded"));
    }
    output.extend_from_slice(&(TAG_MAP.to_le_bytes()));
    output.extend_from_slice(
        &(u64::try_from(values.len()).map_err(|_| invalid("metadata map is too large"))?)
            .to_le_bytes(),
    );
    for (key, value) in values {
        if key.is_empty() {
            return Err(invalid("metadata keys must not be empty"));
        }
        write_string(output, key)?;
        write_value(output, value, depth + 1)?;
    }
    Ok(())
}

fn write_value(output: &mut Vec<u8>, value: &QifV2MetadataValue, depth: usize) -> Result<()> {
    match value {
        QifV2MetadataValue::U32(value) => {
            output.extend_from_slice(&TAG_U32.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::U64(value) => {
            output.extend_from_slice(&TAG_U64.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::I32(value) => {
            output.extend_from_slice(&TAG_I32.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::I64(value) => {
            output.extend_from_slice(&TAG_I64.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::F32(value) => {
            if !value.is_finite() {
                return Err(invalid("metadata F32 values must be finite"));
            }
            output.extend_from_slice(&TAG_F32.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::F64(value) => {
            if !value.is_finite() {
                return Err(invalid("metadata F64 values must be finite"));
            }
            output.extend_from_slice(&TAG_F64.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }
        QifV2MetadataValue::Bool(value) => {
            output.extend_from_slice(&TAG_BOOL.to_le_bytes());
            output.push(u8::from(*value));
        }
        QifV2MetadataValue::String(value) => {
            output.extend_from_slice(&TAG_STRING.to_le_bytes());
            write_string(output, value)?;
        }
        QifV2MetadataValue::Array(values) => {
            if depth > MAX_DEPTH || values.len() > MAX_ITEMS {
                return Err(invalid("metadata nesting or item limit exceeded"));
            }
            if let Some(first) = values.first() {
                let tag = value_tag(first);
                if values.iter().any(|value| value_tag(value) != tag) {
                    return Err(invalid("metadata array values must use one tag"));
                }
            }
            output.extend_from_slice(&TAG_ARRAY.to_le_bytes());
            output.extend_from_slice(
                &u64::try_from(values.len())
                    .map_err(|_| invalid("metadata array is too large"))?
                    .to_le_bytes(),
            );
            for value in values {
                write_value(output, value, depth + 1)?;
            }
        }
        QifV2MetadataValue::Map(values) => write_map(output, values, depth)?,
    }
    Ok(())
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.len() > MAX_STRING_BYTES {
        return Err(invalid("metadata string exceeds the size limit"));
    }
    let length = u64::try_from(value.len()).map_err(|_| invalid("metadata string is too large"))?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn read(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| invalid("metadata cursor overflows"))?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("metadata is truncated"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| invalid("metadata string length overflows"))?;
        if length > MAX_STRING_BYTES {
            return Err(invalid("metadata string exceeds the size limit"));
        }
        let bytes = self.read(length)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| invalid("metadata string is not UTF-8"))
    }
}

fn read_map(cursor: &mut Cursor<'_>, depth: usize) -> Result<BTreeMap<String, QifV2MetadataValue>> {
    if depth > MAX_DEPTH {
        return Err(invalid("metadata nesting limit exceeded"));
    }
    if cursor.u32()? != TAG_MAP {
        return Err(invalid("metadata root must be a map"));
    }
    let count =
        usize::try_from(cursor.u64()?).map_err(|_| invalid("metadata map count overflows"))?;
    if count > MAX_ITEMS {
        return Err(invalid("metadata map item limit exceeded"));
    }
    let mut values = BTreeMap::new();
    for _ in 0..count {
        let key = cursor.string()?;
        if key.is_empty() || values.contains_key(&key) {
            return Err(invalid("metadata keys must be unique and nonempty"));
        }
        values.insert(key, read_value(cursor, depth + 1)?);
    }
    Ok(values)
}

fn read_value(cursor: &mut Cursor<'_>, depth: usize) -> Result<QifV2MetadataValue> {
    if depth > MAX_DEPTH {
        return Err(invalid("metadata nesting limit exceeded"));
    }
    match cursor.u32()? {
        TAG_U32 => Ok(QifV2MetadataValue::U32(cursor.u32()?)),
        TAG_U64 => Ok(QifV2MetadataValue::U64(cursor.u64()?)),
        TAG_I32 => Ok(QifV2MetadataValue::I32(cursor.u32()? as i32)),
        TAG_I64 => Ok(QifV2MetadataValue::I64(cursor.u64()? as i64)),
        TAG_F32 => {
            let value = f32::from_le_bytes(cursor.read(4)?.try_into().unwrap());
            if !value.is_finite() {
                return Err(invalid("metadata F32 value is not finite"));
            }
            Ok(QifV2MetadataValue::F32(value))
        }
        TAG_F64 => {
            let value = f64::from_le_bytes(cursor.read(8)?.try_into().unwrap());
            if !value.is_finite() {
                return Err(invalid("metadata F64 value is not finite"));
            }
            Ok(QifV2MetadataValue::F64(value))
        }
        TAG_BOOL => match cursor.read(1)?[0] {
            0 => Ok(QifV2MetadataValue::Bool(false)),
            1 => Ok(QifV2MetadataValue::Bool(true)),
            _ => Err(invalid("metadata boolean is not 0 or 1")),
        },
        TAG_STRING => Ok(QifV2MetadataValue::String(cursor.string()?)),
        TAG_ARRAY => {
            let count = usize::try_from(cursor.u64()?)
                .map_err(|_| invalid("metadata array count overflows"))?;
            if count > MAX_ITEMS {
                return Err(invalid("metadata array item limit exceeded"));
            }
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_value(cursor, depth + 1)?);
            }
            if let Some(first) = values.first() {
                let tag = value_tag(first);
                if values.iter().any(|value| value_tag(value) != tag) {
                    return Err(invalid("metadata array values must use one tag"));
                }
            }
            Ok(QifV2MetadataValue::Array(values))
        }
        TAG_MAP => {
            cursor.offset = cursor
                .offset
                .checked_sub(4)
                .ok_or_else(|| invalid("metadata cursor underflows"))?;
            Ok(QifV2MetadataValue::Map(read_map(cursor, depth)?))
        }
        tag => Err(invalid(format!("unsupported metadata tag {tag}"))),
    }
}

fn value_tag(value: &QifV2MetadataValue) -> u32 {
    match value {
        QifV2MetadataValue::U32(_) => TAG_U32,
        QifV2MetadataValue::U64(_) => TAG_U64,
        QifV2MetadataValue::I32(_) => TAG_I32,
        QifV2MetadataValue::I64(_) => TAG_I64,
        QifV2MetadataValue::F32(_) => TAG_F32,
        QifV2MetadataValue::F64(_) => TAG_F64,
        QifV2MetadataValue::Bool(_) => TAG_BOOL,
        QifV2MetadataValue::String(_) => TAG_STRING,
        QifV2MetadataValue::Array(_) => TAG_ARRAY,
        QifV2MetadataValue::Map(_) => TAG_MAP,
    }
}
