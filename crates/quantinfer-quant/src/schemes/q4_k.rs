//! Q4_K quantization and dequantization.

use quantinfer_core::{Q4KBlock, QuantizedTensorQ4K, Tensor, Q4_K_BLOCK_SIZE};

use super::validate_finite;
use crate::{QuantizationError, Result};

pub fn quantize_q4_k(input: &Tensor) -> Result<QuantizedTensorQ4K> {
    let values = input.as_f32()?;
    validate_finite(values)?;
    if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q4_K requires a multiple of 256 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q4_K_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q4_K_BLOCK_SIZE>().0 {
        let mut scales = [0.0_f32; 8];
        let mut mins = [0.0_f32; 8];
        let mut quantized = [0_u8; Q4_K_BLOCK_SIZE];
        for (subblock, values) in chunk.as_chunks::<32>().0.iter().enumerate() {
            let min = values
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                .min(0.0);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((max - min) / 15.0).max(f32::EPSILON);
            scales[subblock] = scale;
            mins[subblock] = -min;
            for (index, value) in values.iter().copied().enumerate() {
                quantized[subblock * 32 + index] =
                    ((value - min) / scale).round().clamp(0.0, 15.0) as u8;
            }
        }
        let scale_max = scales.iter().copied().fold(0.0, f32::max).max(f32::EPSILON);
        let min_max = mins.iter().copied().fold(0.0, f32::max);
        let d = scale_max / 63.0;
        let dmin = if min_max == 0.0 { 1.0 } else { min_max / 63.0 };
        let scale_values = scales.map(|value| (value / d).round().clamp(0.0, 63.0) as u8);
        let min_values = mins.map(|value| (value / dmin).round().clamp(0.0, 63.0) as u8);
        let mut packed_scales = [0_u8; 12];
        packed_scales[..4].copy_from_slice(&scale_values[..4]);
        packed_scales[4..8].copy_from_slice(&min_values[..4]);
        for index in 0..4 {
            packed_scales[8 + index] =
                (scale_values[4 + index] & 0x0f) | ((min_values[4 + index] & 0x0f) << 4);
            packed_scales[index] |= (scale_values[4 + index] >> 4) << 6;
            packed_scales[4 + index] |= (min_values[4 + index] >> 4) << 6;
        }
        let mut packed_values = [0_u8; Q4_K_BLOCK_SIZE / 2];
        for group in 0..4 {
            for index in 0..32 {
                let offset = group * 32 + index;
                packed_values[offset] =
                    quantized[group * 64 + index] | (quantized[group * 64 + 32 + index] << 4);
            }
        }
        blocks.push(Q4KBlock {
            scale: d,
            min_scale: dmin,
            scales: packed_scales,
            values: packed_values,
        });
    }
    Ok(QuantizedTensorQ4K::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q4_k(input: &QuantizedTensorQ4K) -> Result<Tensor> {
    let mut output = Vec::with_capacity(input.num_elements());
    for block in input.blocks() {
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..4 {
            scales[index] = block.scales[index] & 0x3f;
            mins[index] = block.scales[4 + index] & 0x3f;
        }
        for index in 0..4 {
            scales[4 + index] =
                (block.scales[8 + index] & 0x0f) | ((block.scales[index] >> 6) << 4);
            mins[4 + index] =
                (block.scales[8 + index] >> 4) | ((block.scales[4 + index] >> 6) << 4);
        }
        for index in 0..Q4_K_BLOCK_SIZE {
            let group = index / 64;
            let within_group = index % 64;
            let packed = block.values[group * 32 + within_group % 32];
            let quantized = if within_group < 32 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let subblock = index / 32;
            let scale = block.scale * f32::from(scales[subblock]);
            let minimum = block.min_scale * f32::from(mins[subblock]);
            output.push(scale * f32::from(quantized) - minimum);
        }
    }
    Ok(Tensor::from_f32(input.shape().clone(), output)?)
}
