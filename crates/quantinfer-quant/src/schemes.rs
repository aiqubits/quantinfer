//! Native quantization and dequantization schemes.

use quantinfer_core::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};

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

pub(crate) fn validate_finite(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(QuantizationError::InvalidInput(
            "quantization requires finite input values".to_owned(),
        ));
    }
    Ok(())
}
