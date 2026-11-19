use super::{invalid, matrix_shape, GptqQuantizationOptions};
use half::f16;
use quantinfer_core::{Q4_0Block, Q8_0Block, QuantizedTensor, QuantizedTensor4, Shape, Tensor};
use quantinfer_quant::{QuantizedData, Result};

/// Static per-output groups use the upstream symmetric min/max rule. Round
/// scales to F16 BEFORE quantization/error feedback, avoiding export drift.
pub(super) fn scales(
    source: &[f32],
    k: usize,
    n: usize,
    options: GptqQuantizationOptions,
    dead: &[bool],
) -> Result<Vec<f32>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(k / 32 * n)
        .map_err(|_| invalid("cannot allocate GPTQ scales"))?;
    for output in 0..n {
        for start in (0..k).step_by(options.group_size) {
            let end = (start + options.group_size.min(k - start)).min(k);
            let mut minimum = 0.0_f64;
            let mut maximum = 0.0_f64;
            for input in start..end {
                let value = if dead[input] {
                    0.0
                } else {
                    f64::from(source[input * n + output])
                };
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            maximum = maximum.max(minimum.abs());
            if minimum < 0.0 {
                minimum = -maximum;
            }
            if maximum == 0.0 {
                minimum = -1.0;
                maximum = 1.0;
            }
            let raw = (maximum - minimum) / ((1_u32 << options.bits) - 1) as f64;
            let scale = f16::from_f64(raw).to_f32();
            if !scale.is_finite() {
                return Err(invalid("GPTQ scale exceeds finite F16 storage range"));
            }
            let scale = if scale == 0.0 {
                f16::from_bits(1).to_f32()
            } else {
                scale
            };
            result.extend(std::iter::repeat_n(scale, (end - start) / 32));
        }
    }
    Ok(result)
}

pub(super) fn pack(
    shape: &Shape,
    codes: &[u8],
    scales: &[f32],
    bits: usize,
) -> Result<QuantizedData> {
    let mut q4 = Vec::new();
    let mut q8 = Vec::new();
    if bits == 4 {
        q4.try_reserve_exact(scales.len())
    } else {
        q8.try_reserve_exact(scales.len())
    }
    .map_err(|_| invalid("cannot allocate GPTQ output blocks"))?;
    for (values, &scale) in codes.as_chunks::<32>().0.iter().zip(scales) {
        if bits == 4 {
            q4.push(Q4_0Block {
                scale,
                values: *values,
            });
        } else {
            q8.push(Q8_0Block {
                scale,
                values: std::array::from_fn(|i| (i16::from(values[i]) - 128) as i8),
            });
        }
    }
    if bits == 4 {
        Ok(QuantizedData::Q4_0(QuantizedTensor4::from_blocks(
            shape.clone(),
            q4,
        )?))
    } else {
        Ok(QuantizedData::Q8_0(QuantizedTensor::from_blocks(
            shape.clone(),
            q8,
        )?))
    }
}

pub(super) fn dequantize(input: &QuantizedData, bits: usize) -> Result<Tensor> {
    let (k, n) = matrix_shape(input.shape())?;
    let mut values = vec![0.0; k * n];
    for output in 0..n {
        for input_index in 0..k {
            let block = output * (k / 32) + input_index / 32;
            let offset = input_index % 32;
            let value = match (input, bits) {
                (QuantizedData::Q4_0(data), 4) => {
                    data.blocks()[block].scale
                        * (f32::from(data.blocks()[block].values[offset]) - 8.0)
                }
                (QuantizedData::Q8_0(data), 8) => {
                    data.blocks()[block].scale * f32::from(data.blocks()[block].values[offset])
                }
                _ => {
                    return Err(invalid(
                        "GPTQ provider expects its signed output-row encoding",
                    ))
                }
            };
            values[input_index * n + output] = value;
        }
    }
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}
