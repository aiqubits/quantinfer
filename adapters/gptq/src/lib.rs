//! Calibrated GPTQ quantization and packed SafeTensors import.
//!
//! Checkpoint imports decode the ordinary affine representation into validated
//! row-major F32 tensors. The calibrated producer instead uses full projection
//! activations to emit native signed Q4_0/Q8_0 output-row blocks, with an explicit
//! storage descriptor and no second quantization.

#![forbid(unsafe_code)]

mod config;
mod decode;
mod quantize;
mod source;

pub use config::GptqConfig;
pub use decode::{dequantize, dequantize_with_zero_point_layout, ZeroPointLayout};
pub use quantize::{GptqProvider, GptqQuantizationOptions, HessianCalibration};
pub use source::GptqTensorSource;
