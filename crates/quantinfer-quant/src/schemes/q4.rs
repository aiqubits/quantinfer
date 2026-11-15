//! Q4_0 quantization and dequantization.

use quantinfer_core::{Q4_0Block, QuantizedTensor4, Tensor, Q4_0_BLOCK_SIZE};

use super::validate_finite;
use crate::{QuantizationError, Result};

pub fn quantize_q4_0(input: &Tensor) -> Result<QuantizedTensor4> {
    let values = input.as_f32()?;
    validate_finite(values)?;
    if !values.len().is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q4_0 requires a multiple of 32 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q4_0_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q4_0_BLOCK_SIZE>().0 {
        let maximum = chunk.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
        let mut quantized = [0_u8; Q4_0_BLOCK_SIZE];
        for (index, value) in chunk.iter().copied().enumerate() {
            quantized[index] = ((value / scale).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
        }
        blocks.push(Q4_0Block {
            scale,
            values: quantized,
        });
    }
    Ok(QuantizedTensor4::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q4_0(input: &QuantizedTensor4) -> Result<Tensor> {
    let values = input
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(move |value| block.scale * (f32::from(*value) - 8.0))
        })
        .collect();
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}
