//! Backend-independent kernel contracts.

#![forbid(unsafe_code)]

use std::fmt;

use quantinfer_core::{
    CoreError, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K,
    QuantizedTensorQ6K, Tensor,
};

pub type Result<T> = std::result::Result<T, KernelError>;

#[derive(Debug, Clone, PartialEq)]
pub enum KernelError {
    InvalidInput(String),
    Core(CoreError),
    Unsupported(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Unsupported(message) => {
                formatter.write_str(message)
            }
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<CoreError> for KernelError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

/// The minimal operator contract required by the first native CPU runtime.
pub trait KernelBackend {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;

    fn matmul_q8_0(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor>;

    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor>;

    fn matmul_q4_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> Result<Tensor>;

    fn matmul_q5_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> Result<Tensor>;

    fn matmul_q6_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> Result<Tensor>;

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor>;

    fn rms_norm_heads(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor>;

    fn softmax(&self, input: &Tensor) -> Result<Tensor>;

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// Apply RoPE. Tables may contain one head-dimension row or one row per token.
    fn apply_rope(&self, input: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Tensor>;

    fn embedding(&self, table: &Tensor, token_ids: &[usize]) -> Result<Tensor>;
}
