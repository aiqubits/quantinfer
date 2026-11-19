//! Compatibility format readers and writers.

#![deny(unsafe_code)]

pub mod dense;
pub mod gguf;
pub mod manifest;
#[path = "qif/mod.rs"]
mod qif_legacy;
mod qif_v2;

/// Stable public QIF API. Wire-version details remain internal.
pub mod qif {
    pub use super::qif_v2::import_gguf;
    pub use super::qif_v2::plan_qif_v2_reshard as plan_qif_reshard;
    pub use super::qif_v2::write_qif_v2 as write_qif;
    pub use super::qif_v2::write_qif_v2_package as write_qif_package;
    pub use super::qif_v2::write_qif_v2_package_with_codec as write_qif_package_with_codec;
    pub use super::qif_v2::write_qif_v2_signed as write_qif_signed;
    pub use super::qif_v2::write_qif_v2_with_codec as write_qif_with_codec;
    pub use super::qif_v2::Codec;
    pub use super::qif_v2::FileReader;
    pub use super::qif_v2::MappedFile;
    pub use super::qif_v2::MetadataValue;
    pub use super::qif_v2::PackageReader;
    pub use super::qif_v2::Reader;
    pub use super::qif_v2::ReshardPlan;
    pub use super::qif_v2::TensorInfo;
    pub use super::qif_v2::{describe_gguf, import_gguf_package, import_gguf_with_codec};
}
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
