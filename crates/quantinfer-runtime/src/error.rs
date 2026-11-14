//! Runtime errors and streaming contracts.

use std::fmt;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    InvalidInput(String),
    Core(quantinfer_core::CoreError),
    Format(quantinfer_format::FormatError),
    Kernel(quantinfer_kernel::KernelError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamControl {
    Continue,
    Stop,
}

pub trait TokenSink {
    fn on_token(&mut self, token_id: usize) -> Result<StreamControl>;
}

impl<F> TokenSink for F
where
    F: FnMut(usize) -> Result<StreamControl>,
{
    fn on_token(&mut self, token_id: usize) -> Result<StreamControl> {
        self(token_id)
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Core(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Kernel(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<quantinfer_core::CoreError> for RuntimeError {
    fn from(error: quantinfer_core::CoreError) -> Self {
        Self::Core(error)
    }
}

impl From<quantinfer_format::FormatError> for RuntimeError {
    fn from(error: quantinfer_format::FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<quantinfer_kernel::KernelError> for RuntimeError {
    fn from(error: quantinfer_kernel::KernelError) -> Self {
        Self::Kernel(error)
    }
}
