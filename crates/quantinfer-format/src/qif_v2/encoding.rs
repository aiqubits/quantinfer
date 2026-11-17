//! Validation of the fixed physical encodings in the current wire profile.

use quantinfer_core::{
    DType, QuantizationAxis, ScaleType, TensorStorageDescriptor, ZeroPointType, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};

use super::{invalid, MAX_RANK};
use crate::Result;

pub(super) fn validate_descriptor(descriptor: &TensorStorageDescriptor) -> Result<()> {
    descriptor.validate()?;
    if descriptor.logical_shape.dims().len() > MAX_RANK {
        return Err(invalid("tensor rank exceeds the limit"));
    }
    let (block_size, packing) = match descriptor.dtype {
        DType::F32 | DType::F16 | DType::BF16 => (None, "none"),
        DType::Q8_0 => (Some(Q8_0_BLOCK_SIZE), "scalar"),
        DType::Q4_0 => (Some(Q4_0_BLOCK_SIZE), "nibble_split_half"),
        DType::Q4_K | DType::Q5_K | DType::Q6_K => (Some(Q4_K_BLOCK_SIZE), "k_quant"),
    };
    let quantized = block_size.is_some();
    let axis_valid = match (&descriptor.quantization_axis, quantized) {
        (QuantizationAxis::None, false) | (QuantizationAxis::Tensor, true) => true,
        (QuantizationAxis::Axis(0), true) => {
            descriptor.physical_layout == crate::GGML_OUTPUT_ROWS_LAYOUT
                && descriptor.logical_shape.dims().len() == 2
        }
        _ => false,
    };
    if descriptor.block_size != block_size
        || descriptor.group_size.is_some()
        || descriptor.packing != packing
        || descriptor.scale_type != quantized.then_some(ScaleType::F16)
        || descriptor.zero_point_type != ZeroPointType::None
        || !axis_valid
    {
        return Err(invalid(
            "storage descriptor disagrees with physical encoding",
        ));
    }
    Ok(())
}
