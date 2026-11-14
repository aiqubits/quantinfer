//! SafeTensors payload decoding.

use half::{bf16, f16};

use super::types::{SafeTensorDType, SafeTensorInfo};
use crate::{FormatError, Result};

pub(crate) fn decode_tensor_f32(
    info: &SafeTensorInfo,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    let values: Vec<f32> = match info.dtype {
        SafeTensorDType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        SafeTensorDType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
            .collect(),
        SafeTensorDType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
            .collect(),
        SafeTensorDType::I32
        | SafeTensorDType::U32
        | SafeTensorDType::I64
        | SafeTensorDType::U64 => {
            return Err(FormatError::Invalid(format!(
                "tensor {name} has integer dtype {:?}, expected a floating dtype",
                info.dtype
            )))
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} contains non-finite values"
        )));
    }
    Ok(values)
}

pub(crate) fn decode_tensor_u32(
    info: &SafeTensorInfo,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    match info.dtype {
        SafeTensorDType::I32 | SafeTensorDType::U32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()),
        dtype => Err(FormatError::Invalid(format!(
            "tensor {name} has dtype {dtype:?}, expected I32 or U32"
        ))),
    }
}

pub(crate) fn decode_tensor_u64(
    info: &SafeTensorInfo,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    match info.dtype {
        SafeTensorDType::I64 | SafeTensorDType::U64 => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_le_bytes(*chunk))
            .collect()),
        dtype => Err(FormatError::Invalid(format!(
            "tensor {name} has dtype {dtype:?}, expected I64 or U64"
        ))),
    }
}
