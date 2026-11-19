//! Internal current QIF inference weight container implementation.
//!
//! This module is intentionally independent from the retired QIF v1 wire
//! implementation in a private legacy module. QIF has a fixed binary index and
//! chunk table so a file-backed reader can locate tensors without scanning
//! payload bytes.

mod backing;
mod chunk;
mod codec;
mod encoding;
mod metadata;
mod mmap;
mod package;
mod payload;
mod reader;
mod reshard;
mod signature;
mod writer;

pub use metadata::MetadataValue;
pub use mmap::MappedFile;
pub use package::{write_qif_v2_package, write_qif_v2_package_with_codec, PackageReader};
pub use reader::{FileReader, Reader, TensorInfo};
pub use reshard::{plan_qif_v2_reshard, ReshardPlan};
pub use writer::{write_qif_v2, write_qif_v2_signed, write_qif_v2_with_codec, Codec};

pub(crate) use backing::{open_packed_file, open_packed_path, FileBacking};
pub(crate) use reader::has_magic;
pub(crate) use reader::open_packed;
pub use writer::{describe_gguf, import_gguf, import_gguf_package, import_gguf_with_codec};

use std::collections::BTreeMap;

use crate::{FormatError, GgufValue, ModelManifest, Result};

pub(crate) const MAGIC: &[u8; 4] = b"QIF\0";
pub(crate) const VERSION: u32 = 2;
pub(crate) const HEADER_SIZE: usize = 144;
pub(crate) const INDEX_MAGIC: &[u8; 4] = b"QIDX";
pub(crate) const INDEX_VERSION: u32 = 1;
pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const MAX_HEADER_SECTION: usize = 64 * 1024 * 1024;
pub(crate) const MAX_JSON_DEPTH: usize = 32;
pub(crate) const MAX_INDEX_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const MAX_TENSORS: usize = 1_048_576;
pub(crate) const MAX_CHUNKS: usize = 4_194_304;
pub(crate) const MAX_RANK: usize = 32;
pub(crate) const HASH_SIZE: usize = 32;
pub(crate) const HASH_SHA256: u32 = 1;
pub(crate) const CODEC_NONE: u32 = 0;
pub(crate) const CODEC_ZSTD: u32 = 1;
pub(crate) const CODEC_LZ4: u32 = 2;
pub(crate) const ALIAS_NONE: u64 = u64::MAX;
pub(crate) const SIGNATURE_LENGTH: usize = signature::LENGTH;

pub(crate) fn canonicalize_manifest_layouts(manifest: &mut ModelManifest) {
    for tensor in &mut manifest.tensors {
        if tensor.descriptor.physical_layout == "output_rows" {
            tensor.descriptor.physical_layout = crate::GGML_OUTPUT_ROWS_LAYOUT.to_owned();
        }
    }
}

pub(crate) fn invalid(message: impl std::fmt::Display) -> FormatError {
    FormatError::Invalid(format!("QIF {message}"))
}

pub(super) fn bytes_with_capacity(length: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| invalid(format!("cannot allocate {length} bytes: {error}")))?;
    Ok(bytes)
}

pub(super) fn zeroed_bytes(length: usize) -> Result<Vec<u8>> {
    let mut bytes = bytes_with_capacity(length)?;
    bytes.resize(length, 0);
    Ok(bytes)
}

pub(super) fn copy_bytes(source: &[u8]) -> Result<Vec<u8>> {
    let mut bytes = bytes_with_capacity(source.len())?;
    bytes.extend_from_slice(source);
    Ok(bytes)
}

/// Reject excessively nested JSON before serde recursively builds a value.
///
/// This is a lightweight preflight bound for untrusted manifests. Full JSON
/// syntax, duplicate keys, and canonical serialization are still validated by
/// serde and the typed manifest parser afterwards.
pub(crate) fn validate_json_depth(bytes: &[u8]) -> Result<()> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth
                    .checked_add(1)
                    .ok_or_else(|| invalid("JSON nesting depth overflows"))?;
                if depth > MAX_JSON_DEPTH {
                    return Err(invalid("JSON nesting depth exceeds the limit"));
                }
            }
            b'}' | b']' => {
                if depth == 0 {
                    return Err(invalid("JSON nesting is unbalanced"));
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    if in_string || depth != 0 {
        return Err(invalid("JSON nesting is unbalanced"));
    }
    Ok(())
}

pub(crate) fn align(offset: usize) -> Result<usize> {
    offset
        .checked_next_multiple_of(ALIGNMENT)
        .ok_or_else(|| invalid("offset alignment overflows usize"))
}

pub(crate) fn checked_range(bytes: &[u8], offset: u64, length: u64) -> Result<&[u8]> {
    let start = usize::try_from(offset).map_err(|_| invalid("offset overflows usize"))?;
    let length = usize::try_from(length).map_err(|_| invalid("length overflows usize"))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| invalid("range overflows usize"))?;
    bytes
        .get(start..end)
        .ok_or_else(|| invalid("range exceeds file length"))
}

pub(crate) fn metadata_from_gguf(
    values: &BTreeMap<String, GgufValue>,
) -> BTreeMap<String, MetadataValue> {
    values
        .iter()
        .map(|(key, value)| (key.clone(), MetadataValue::from(value)))
        .collect()
}

pub(crate) fn metadata_to_gguf(
    values: &BTreeMap<String, MetadataValue>,
) -> Result<BTreeMap<String, GgufValue>> {
    values
        .iter()
        .filter(|(_, value)| value.can_project_to_gguf())
        .map(|(key, value)| Ok((key.clone(), value.to_gguf()?)))
        .collect()
}

#[cfg(test)]
mod tests;
