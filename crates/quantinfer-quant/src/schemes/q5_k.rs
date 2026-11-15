//! Q5_K quantization and dequantization.

use quantinfer_core::{Q5KBlock, QuantizedTensorQ5K, Tensor, Q4_K_BLOCK_SIZE};

use super::validate_finite;
use crate::{QuantizationError, Result};

pub fn quantize_q5_k(input: &Tensor) -> Result<QuantizedTensorQ5K> {
    let values = input.as_f32()?;
    validate_finite(values)?;
    if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q5_K requires a multiple of 256 elements".to_owned(),
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
            let scale = ((max - min) / 31.0).max(f32::EPSILON);
            scales[subblock] = scale;
            mins[subblock] = -min;
            for (index, value) in values.iter().copied().enumerate() {
                quantized[subblock * 32 + index] =
                    ((value - min) / scale).round().clamp(0.0, 31.0) as u8;
            }
        }
        let scale_max = scales.iter().copied().fold(0.0, f32::max).max(f32::EPSILON);
        let min_max = mins.iter().copied().fold(0.0, f32::max);
        let d = scale_max / 63.0;
        let dmin = if min_max == 0.0 { 1.0 } else { min_max / 63.0 };
        let sv = scales.map(|value| (value / d).round().clamp(0.0, 63.0) as u8);
        let mv = mins.map(|value| (value / dmin).round().clamp(0.0, 63.0) as u8);
        let mut packed = [0_u8; 12];
        packed[..4].copy_from_slice(&sv[..4]);
        packed[4..8].copy_from_slice(&mv[..4]);
        for index in 0..4 {
            packed[8 + index] = (sv[4 + index] & 0xf) | ((mv[4 + index] & 0xf) << 4);
            packed[index] |= (sv[4 + index] >> 4) << 6;
            packed[4 + index] |= (mv[4 + index] >> 4) << 6;
        }
        let mut low = [0_u8; 128];
        let mut high = [0_u8; 32];
        for group in 0..4 {
            for index in 0..32 {
                let offset = group * 32 + index;
                let first = quantized[group * 64 + index];
                let second = quantized[group * 64 + 32 + index];
                low[offset] = (first & 0xf) | ((second & 0xf) << 4);
                if first & 0x10 != 0 {
                    high[index] |= 1 << (2 * group);
                }
                if second & 0x10 != 0 {
                    high[index] |= 1 << (2 * group + 1);
                }
            }
        }
        blocks.push(Q5KBlock {
            scale: d,
            min_scale: dmin,
            scales: packed,
            high_bits: high,
            values: low,
        });
    }
    Ok(QuantizedTensorQ5K::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q5_k(input: &QuantizedTensorQ5K) -> Result<Tensor> {
    let mut output = Vec::with_capacity(input.num_elements());
    for block in input.blocks() {
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..4 {
            scales[index] = block.scales[index] & 0x3f;
            mins[index] = block.scales[4 + index] & 0x3f;
            scales[4 + index] = (block.scales[8 + index] & 0xf) | ((block.scales[index] >> 6) << 4);
            mins[4 + index] =
                (block.scales[8 + index] >> 4) | ((block.scales[4 + index] >> 6) << 4);
        }
        for index in 0..Q4_K_BLOCK_SIZE {
            let group = index / 64;
            let within = index % 64;
            let packed = block.values[group * 32 + within % 32];
            let mut q = if within < 32 {
                packed & 0xf
            } else {
                packed >> 4
            };
            if block.high_bits[within % 32]
                & (1 << (if within < 32 {
                    2 * group
                } else {
                    2 * group + 1
                }))
                != 0
            {
                q += 16;
            }
            let subblock = index / 32;
            output.push(
                block.scale * f32::from(scales[subblock]) * f32::from(q)
                    - block.min_scale * f32::from(mins[subblock]),
            );
        }
    }
    Ok(Tensor::from_f32(input.shape().clone(), output)?)
}
