//! Shared GGUF types and format errors.

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;

use super::Result;

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
pub(crate) struct PendingTensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: GgmlType,
    pub(crate) data: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct GgufWriter {
    pub(crate) metadata: BTreeMap<String, GgufValue>,
    pub(crate) tensors: Vec<PendingTensor>,
    pub(crate) alignment: u64,
}
