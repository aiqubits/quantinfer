//! Compatibility format readers and writers.

#![deny(unsafe_code)]

pub mod dense;
pub mod gguf;
pub mod manifest;
pub mod qif;
pub mod qif_v2;
pub mod safetensors;

mod json;
pub(crate) use json::DuplicateFreeJson;

pub use dense::{
    canonical_dense_tensor_name, is_embedding_storage_name, is_linear_projection_storage_name,
    is_lm_head_storage_name, is_matrix_storage_name, transpose_dense_layout,
};
pub use gguf::*;
pub(crate) use gguf::{encode_value, tensor_byte_size, write_string, Cursor};
pub use manifest::*;
pub use safetensors::{
    SafeTensorDType, SafeTensorFile, SafeTensorFileReader, SafeTensorInfo, SafeTensorShards,
};
