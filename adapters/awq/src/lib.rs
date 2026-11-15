//! AutoAWQ SafeTensors importer.
//!
//! AutoAWQ stores 4-bit values packed along the output dimension and uses an
//! interleaved bit order inside each 32-bit word. This adapter decodes that
//! representation into a row-major F32 tensor at the format boundary.

#![forbid(unsafe_code)]

mod config;
mod decode;
mod source;

pub use config::AwqConfig;
pub use decode::dequantize;
pub use source::AwqTensorSource;
