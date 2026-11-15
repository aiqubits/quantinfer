//! LLM Compressor `compressed-tensors` WNA16 importer.
//!
//! The adapter handles the integer packed weight-only layout used by
//! compressed-tensors: `weight_packed` is packed along the input axis,
//! `weight_scale` is indexed by output and input group, and the optional
//! `weight_zero_point` is packed along the output axis.

#![forbid(unsafe_code)]

mod config;
mod decode;
mod source;

pub use config::LlmCompressorConfig;
pub use decode::dequantize;
pub use source::LlmCompressorTensorSource;
