//! Core error types.

use std::fmt;

use crate::DType;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidShape(String),
    InvalidData(String),
    DTypeMismatch { expected: DType, actual: DType },
    Unsupported(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message)
            | Self::InvalidData(message)
            | Self::Unsupported(message) => formatter.write_str(message),
            Self::DTypeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dtype mismatch: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for CoreError {}
