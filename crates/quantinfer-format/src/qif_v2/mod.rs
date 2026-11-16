//! QIF v2 inference weight container.
//!
//! This module is intentionally independent from the retired QIF v1 wire
//! implementation in [`crate::qif`]. QIF v2 has a fixed binary index and
//! chunk table so a file-backed reader can locate tensors without scanning
//! payload bytes.

mod codec;
mod metadata;
mod mmap;
mod package;
mod reader;
mod signature;
mod writer;

pub use metadata::QifV2MetadataValue;
pub use mmap::QifV2MappedFile;
pub use package::{write_qif_v2_package, write_qif_v2_package_with_codec, QifV2PackageReader};
pub use reader::{QifV2FileReader, QifV2Reader, QifV2TensorInfo};
pub use writer::{write_qif_v2, write_qif_v2_signed, write_qif_v2_with_codec, QifV2Codec};

pub(crate) use reader::has_magic;
pub(crate) use reader::open_packed;
pub use writer::import_gguf;

use std::collections::BTreeMap;

use crate::{FormatError, GgufValue, ModelManifest, Result};

pub(crate) const MAGIC: &[u8; 4] = b"QIF\0";
pub(crate) const VERSION: u32 = 2;
pub(crate) const HEADER_SIZE: usize = 144;
pub(crate) const INDEX_MAGIC: &[u8; 4] = b"QIDX";
pub(crate) const INDEX_VERSION: u32 = 1;
pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const MAX_HEADER_SECTION: usize = 64 * 1024 * 1024;
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
    FormatError::Invalid(format!("QIF v2 {message}"))
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
) -> BTreeMap<String, QifV2MetadataValue> {
    values
        .iter()
        .map(|(key, value)| (key.clone(), QifV2MetadataValue::from(value)))
        .collect()
}

pub(crate) fn metadata_to_gguf(
    values: &BTreeMap<String, QifV2MetadataValue>,
) -> Result<BTreeMap<String, GgufValue>> {
    values
        .iter()
        .map(|(key, value)| Ok((key.clone(), value.to_gguf()?)))
        .collect()
}

#[cfg(test)]
mod tests;
