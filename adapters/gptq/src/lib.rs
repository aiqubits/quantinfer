//! GPTQ SafeTensors importer.
//!
//! The importer decodes the ordinary affine GPTQ representation into a
//! row-major F32 tensor. It keeps the packed representation at the format
//! boundary so runtimes can choose their own execution layout.

#![forbid(unsafe_code)]

mod config;
mod decode;
mod source;

pub use config::GptqConfig;
pub use decode::{dequantize, dequantize_with_zero_point_layout, ZeroPointLayout};
pub use source::GptqTensorSource;
