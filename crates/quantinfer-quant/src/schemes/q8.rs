//! Q8_0 quantization and dequantization.

use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor, Q8_0_BLOCK_SIZE};

use super::validate_finite;
use crate::{QuantizationError, Result};

pub fn quantize_q8_0(input: &Tensor) -> Result<QuantizedTensor> {
    let values = input.as_f32()?;
    validate_finite(values)?;
    if !values.len().is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q8_0 requires a multiple of 32 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q8_0_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q8_0_BLOCK_SIZE>().0 {
        let maximum = chunk.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        let mut quantized = [0_i8; Q8_0_BLOCK_SIZE];
        for (index, value) in chunk.iter().copied().enumerate() {
            let rounded = (value / scale).round().clamp(-127.0, 127.0);
            quantized[index] = rounded as i8;
        }
        blocks.push(Q8_0Block {
            scale,
            values: quantized,
        });
    }
    Ok(QuantizedTensor::from_blocks(input.shape().clone(), blocks)?)
}

pub fn dequantize_q8_0(input: &QuantizedTensor) -> Result<Tensor> {
    let values = input
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(move |value| block.scale * f32::from(*value))
        })
        .collect();
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}
