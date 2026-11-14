//! Compatibility format readers and writers.

#![forbid(unsafe_code)]

pub mod gguf;
pub mod manifest;
pub mod qif;
pub mod safetensors;

pub use gguf::*;
pub(crate) use gguf::{encode_value, tensor_byte_size, write_string, Cursor};
pub use manifest::*;
pub use safetensors::{
    SafeTensorDType, SafeTensorFile, SafeTensorFileReader, SafeTensorInfo, SafeTensorShards,
};
