//! Q6_K quantization and dequantization.

use quantinfer_core::{Q6KBlock, QuantizedTensorQ6K, Tensor, Q4_K_BLOCK_SIZE};

use super::validate_finite;
use crate::{QuantizationError, Result};

pub fn quantize_q6_k(input: &Tensor) -> Result<QuantizedTensorQ6K> {
    let values = input.as_f32()?;
    validate_finite(values)?;
    if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q6_K requires a multiple of 256 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q4_K_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q4_K_BLOCK_SIZE>().0 {
        let mut scales_f = [0.0_f32; 16];
        let mut quantized = [0_u8; Q4_K_BLOCK_SIZE];
        for (subblock, values) in chunk.as_chunks::<16>().0.iter().enumerate() {
            let max_abs = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = (max_abs / 31.0).max(f32::EPSILON);
            scales_f[subblock] = scale;
            for (index, value) in values.iter().copied().enumerate() {
                quantized[subblock * 16 + index] =
                    ((value / scale).round().clamp(-32.0, 31.0) as i32 + 32) as u8;
            }
        }
        let max_scale = scales_f
            .iter()
            .copied()
            .fold(0.0, f32::max)
            .max(f32::EPSILON);
        let d = max_scale / 127.0;
        let scales = scales_f.map(|value| (value / d).round().clamp(-128.0, 127.0) as i8);
        let mut low_bits = [0_u8; 128];
        let mut high_bits = [0_u8; 64];
        for ip in 0..2 {
            for il in 0..32 {
                let q0 = quantized[ip * 128 + il];
                let q1 = quantized[ip * 128 + 32 + il];
                let q2 = quantized[ip * 128 + 64 + il];
                let q3 = quantized[ip * 128 + 96 + il];
                low_bits[ip * 64 + il] = (q0 & 0xf) | ((q2 & 0xf) << 4);
                low_bits[ip * 64 + 32 + il] = (q1 & 0xf) | ((q3 & 0xf) << 4);
                high_bits[ip * 32 + il] = ((q0 >> 4) & 3)
                    | (((q1 >> 4) & 3) << 2)
                    | (((q2 >> 4) & 3) << 4)
                    | (((q3 >> 4) & 3) << 6);
            }
        }
        blocks.push(Q6KBlock {
            scale: d,
            scales,
            low_bits,
            high_bits,
        });
    }
    Ok(QuantizedTensorQ6K::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q6_k(input: &QuantizedTensorQ6K) -> Result<Tensor> {
    let mut output = Vec::with_capacity(input.num_elements());
    for block in input.blocks() {
        let mut restored = vec![0.0_f32; Q4_K_BLOCK_SIZE];
        for ip in 0..2 {
            for il in 0..32 {
                let high = block.high_bits[ip * 32 + il];
                let q0 = (block.low_bits[ip * 64 + il] & 0xf) | ((high & 3) << 4);
                let q1 = (block.low_bits[ip * 64 + 32 + il] & 0xf) | (((high >> 2) & 3) << 4);
                let q2 = (block.low_bits[ip * 64 + il] >> 4) | (((high >> 4) & 3) << 4);
                let q3 = (block.low_bits[ip * 64 + 32 + il] >> 4) | (((high >> 6) & 3) << 4);
                let bases = [
                    ip * 128 + il,
                    ip * 128 + 32 + il,
                    ip * 128 + 64 + il,
                    ip * 128 + 96 + il,
                ];
                let qs = [q0, q1, q2, q3];
                for (offset, q) in bases.into_iter().zip(qs) {
                    let subblock = offset / 16;
                    let signed = i32::from(q) - 32;
                    restored[offset] =
                        block.scale * f32::from(block.scales[subblock]) * signed as f32;
                }
            }
        }
        output.extend_from_slice(&restored);
    }
    Ok(Tensor::from_f32(input.shape().clone(), output)?)
}
