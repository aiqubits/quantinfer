//! Lossless packed storage with an explicit QIF logical-coordinate view.

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{Device, Tensor as CandleTensor};
use quantinfer_core::DType;
use quantinfer_format::qif::{MetadataValue, Reader as QifReader};

use super::{InteropError, Result};

/// Original quantized blocks plus their QIF logical coordinates.
///
/// Native output-row matrices have physical shape [N,K] for logical [K,N].
/// Candle's QTensor cannot carry a transpose stride. Retain the blocks in
/// physical order and apply the coordinate view when decoding instead.
pub struct PackedQifTensor {
    storage: QTensor,
    logical_shape: Vec<usize>,
    transposed: bool,
}

impl PackedQifTensor {
    /// Physical packed storage. Inspect `is_transposed` before using its shape
    /// as QIF logical coordinates. Codes and scales have not been requantized.
    pub fn storage(&self) -> &QTensor {
        &self.storage
    }

    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    pub fn is_transposed(&self) -> bool {
        self.transposed
    }

    /// Decode in QIF logical shape/element order, preserving Candle strides.
    pub fn dequantize(&self, device: &Device) -> Result<CandleTensor> {
        let decoded = self
            .storage
            .dequantize(device)
            .map_err(InteropError::Candle)?;
        if self.transposed {
            decoded.transpose(0, 1).map_err(InteropError::Candle)
        } else {
            Ok(decoded)
        }
    }
}

/// Import a packed QIF tensor with its original logical shape.
///
/// A plain QTensor cannot represent transposed packed coordinates. Use
/// `packed_view_from_qif` for native output-row matrices; this entry point
/// rejects those layouts rather than changing the logical weight values.
pub fn packed_from_qif(reader: &QifReader<'_>, name: &str, device: &Device) -> Result<QTensor> {
    let view = packed_view_from_qif(reader, name, device)?;
    if view.transposed {
        return Err(InteropError::Invalid(
            "QIF output-row coordinates need packed_view_from_qif; a plain Candle QTensor cannot preserve the logical shape".into(),
        ));
    }
    Ok(view.storage)
}

/// Verify the selected tensor and import its blocks without requantization.
///
/// Supports independent none/zstd/LZ4 chunks and aliases. Verification covers
/// this tensor's raw hash and finite values; it is not full-artifact verify.
pub fn packed_view_from_qif(
    reader: &QifReader<'_>,
    name: &str,
    device: &Device,
) -> Result<PackedQifTensor> {
    if cfg!(target_endian = "big") {
        return Err(InteropError::Invalid(
            "packed Candle interchange requires a little-endian target".into(),
        ));
    }
    let tensor = reader
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| InteropError::Invalid(format!("unknown QIF tensor {name}")))?;
    let descriptor = &tensor.descriptor;
    let dtype = match descriptor.dtype {
        DType::Q8_0 => GgmlDType::Q8_0,
        DType::Q4_0 => GgmlDType::Q4_0,
        DType::Q4_K => GgmlDType::Q4K,
        DType::Q5_K => GgmlDType::Q5K,
        DType::Q6_K => GgmlDType::Q6K,
        _ => {
            return Err(InteropError::Invalid(
                "QIF tensor does not use a supported packed quantization".into(),
            ))
        }
    };
    let logical_shape = descriptor.logical_shape.dims().to_vec();
    let output_rows = descriptor.physical_layout == quantinfer_format::GGML_OUTPUT_ROWS_LAYOUT;
    if output_rows && logical_shape.len() > 2 {
        return Err(InteropError::Invalid(
            "QIF output-row coordinates are defined for matrices and continuous vectors".into(),
        ));
    }
    let ggml_dimensions = matches!(
        reader.metadata().get("quantinfer.qif.ggml_dimension_order"),
        Some(MetadataValue::Bool(true))
    );
    let transposed = output_rows && logical_shape.len() == 2 && !ggml_dimensions;
    let mut physical_shape = logical_shape.clone();
    if transposed {
        physical_shape.reverse();
    }
    if physical_shape
        .last()
        .is_none_or(|width| !width.is_multiple_of(dtype.block_size()))
    {
        return Err(InteropError::Invalid(
            "Candle requires quantization blocks to fit within each physical tensor row".into(),
        ));
    }
    let bytes = reader.tensor_data(name).map_err(InteropError::Format)?;
    let expected_hash = tensor
        .checksum_sha256
        .as_deref()
        .ok_or_else(|| InteropError::Invalid("QIF tensor checksum is missing".into()))?;
    if !quantinfer_format::sha256_hex(&bytes).eq_ignore_ascii_case(expected_hash) {
        return Err(InteropError::Invalid(format!(
            "QIF tensor {name} checksum mismatch"
        )));
    }
    let expected = descriptor
        .logical_shape
        .num_elements()
        .map_err(InteropError::Core)?
        / dtype.block_size()
        * dtype.type_size();
    if bytes.len() != expected {
        return Err(InteropError::Invalid(
            "QIF and Candle block sizes disagree".into(),
        ));
    }
    // Candle's loader borrows aligned native blocks while copying to storage.
    // Do not assume the allocation's u8 alignment satisfies a u16 block.
    let aligned;
    let data = if bytes.as_ptr().cast::<u16>().is_aligned() {
        bytes.as_slice()
    } else {
        aligned = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_ne_bytes(*pair))
            .collect::<Vec<_>>();
        bytemuck::cast_slice(&aligned)
    };
    let storage = QStorage::from_data(Cow::Borrowed(data), device, dtype)
        .and_then(|storage| QTensor::new(storage, physical_shape))
        .map_err(InteropError::Candle)?;
    Ok(PackedQifTensor {
        storage,
        logical_shape,
        transposed,
    })
}

const _: () = {
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8_0};
    assert!(align_of::<BlockQ4_0>() == align_of::<u16>());
    assert!(align_of::<BlockQ8_0>() == align_of::<u16>());
    assert!(align_of::<BlockQ4K>() == align_of::<u16>());
    assert!(align_of::<BlockQ5K>() == align_of::<u16>());
    assert!(align_of::<BlockQ6K>() == align_of::<u16>());
};

#[cfg(test)]
mod tests;
