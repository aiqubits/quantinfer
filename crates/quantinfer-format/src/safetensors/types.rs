//! SafeTensors type metadata.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    I64,
    U64,
}

impl SafeTensorDType {
    pub(crate) fn parse(value: &str) -> crate::Result<Self> {
        match value {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I32" => Ok(Self::I32),
            "U32" => Ok(Self::U32),
            "I64" => Ok(Self::I64),
            "U64" => Ok(Self::U64),
            _ => Err(crate::FormatError::Invalid(format!(
                "unsupported SafeTensors dtype {value}"
            ))),
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Keep hostile JSON headers bounded before allocating their representation.
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
