//! Versioned QuantInfer model containers.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write;
use std::ops::Range;

use quantinfer_core::{
    DType, QuantizationScheme, Shape, TensorStorageDescriptor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
};
use sha2::{Digest, Sha256};

use crate::{
    encode_value, sha256_hex, tensor_byte_size, write_string, Cursor, FormatError, GgmlType,
    GgufReader, GgufTensorInfo, GgufValue, ModelManifest, PackedTensorReader, Result,
    TensorManifest,
};

const MAGIC: &[u8; 4] = b"QIF\0";
const VERSION: u32 = 1;
const PREFIX: usize = 56;
const ALIGNMENT: usize = 64;
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARRAY_ITEMS: usize = 1_048_576;
const GGML_DIMENSION_ORDER: &str = "quantinfer.qif.ggml_dimension_order";

fn payload_size(descriptor: &TensorStorageDescriptor) -> Result<usize> {
    let (encoding, scheme, block_size) = storage_encoding(descriptor.dtype);
    let shape = descriptor.logical_shape.clone();
    let expected = if scheme == QuantizationScheme::None {
        TensorStorageDescriptor::floating(descriptor.dtype, shape)?
    } else {
        TensorStorageDescriptor::quantized(descriptor.dtype, scheme, shape, block_size)?
    };
    if descriptor != &expected {
        return Err(invalid("unsupported tensor storage descriptor"));
    }
    tensor_byte_size(encoding, descriptor.logical_shape.dims())
}

fn storage_encoding(dtype: DType) -> (GgmlType, QuantizationScheme, usize) {
    match dtype {
        DType::F32 => (GgmlType::F32, QuantizationScheme::None, 0),
        DType::F16 => (GgmlType::F16, QuantizationScheme::None, 0),
        DType::BF16 => (GgmlType::BF16, QuantizationScheme::None, 0),
        DType::Q8_0 => (GgmlType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
        DType::Q4_0 => (GgmlType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
        DType::Q4_K => (GgmlType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
        DType::Q5_K => (GgmlType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
        DType::Q6_K => (GgmlType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
    }
}

fn read_metadata(bytes: &[u8]) -> Result<BTreeMap<String, GgufValue>> {
    let mut cursor = Cursor::new(bytes);
    let count = cursor.read_u64()?;
    if count > (bytes.len() / 12) as u64 {
        return Err(invalid("metadata count exceeds its section"));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = cursor.read_string()?;
        let value = cursor.read_value_with_limit(MAX_ARRAY_ITEMS)?;
        if key.is_empty() || metadata.insert(key, value).is_some() {
            return Err(invalid("metadata keys must be unique and nonempty"));
        }
    }
    if cursor.position() != bytes.len() {
        return Err(invalid("unexpected bytes after metadata"));
    }
    Ok(metadata)
}

fn write_metadata(metadata: &BTreeMap<String, GgufValue>) -> Result<Vec<u8>> {
    let mut bytes = (metadata.len() as u64).to_le_bytes().to_vec();
    for (key, value) in metadata {
        if key.is_empty() {
            return Err(invalid("metadata keys must be nonempty"));
        }
        if matches!(value, GgufValue::Array(values) if values.len() > MAX_ARRAY_ITEMS) {
            return Err(invalid("metadata array exceeds the item limit"));
        }
        write_string(&mut bytes, key)?;
        if matches!(value, GgufValue::Array(values) if values.is_empty()) {
            bytes.extend_from_slice(&9_u32.to_le_bytes());
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
        } else {
            encode_value(&mut bytes, value)?;
        }
    }
    Ok(bytes)
}

fn header_size(bytes: &[u8]) -> Result<usize> {
    let size = u64::from_le_bytes(bytes.try_into().unwrap());
    if size > MAX_HEADER_BYTES as u64 {
        return Err(invalid("header section exceeds the size limit"));
    }
    Ok(size as usize)
}

fn aligned(offset: usize) -> Result<usize> {
    offset
        .checked_next_multiple_of(ALIGNMENT)
        .ok_or_else(|| invalid("tensor alignment overflows usize"))
}

fn check_padding(bytes: &[u8], range: Range<usize>) -> Result<()> {
    if !bytes
        .get(range)
        .is_some_and(|padding| padding.iter().all(|&byte| byte == 0))
    {
        return Err(invalid("missing or nonzero tensor padding"));
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> FormatError {
    FormatError::Invalid(format!("QIF {message}"))
}

fn write_error(error: std::io::Error) -> FormatError {
    FormatError::Io(format!("cannot write QIF: {error}"))
}

mod reader;
mod transcode;
mod writer;

#[cfg(test)]
mod tests;

pub use reader::QifReader;
pub(crate) use reader::{has_magic, open_packed, uses_ggml_dimension_order};
pub use transcode::import_gguf;
pub use writer::write_qif;
