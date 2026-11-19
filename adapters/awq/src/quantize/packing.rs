use super::{invalid, matrix_shape, transforms::validate_scales, AwqOptions};
use half::f16;
use quantinfer_core::{Q4_0Block, Q8_0Block, QuantizedTensor, QuantizedTensor4, Tensor};
use quantinfer_quant::{QuantizedData, Result};

pub(super) fn quantize(
    weight: &Tensor,
    scales: Option<&[f32]>,
    options: AwqOptions,
) -> Result<QuantizedData> {
    let (k, n) = matrix_shape(weight.shape())?;
    let values = weight.as_f32()?;
    if let Some(scales) = scales {
        validate_scales(scales, k)?;
    }
    let mut q4 = Vec::new();
    let mut q8 = Vec::new();
    let count = k / 32 * n;
    if options.bits == 4 {
        q4.try_reserve_exact(count)
    } else {
        q8.try_reserve_exact(count)
    }
    .map_err(|_| invalid("cannot allocate AWQ blocks"))?;
    for output in 0..n {
        for start in (0..k).step_by(options.group_size) {
            let end = start + options.group_size.min(k - start);
            let value = |input: usize| values[input * n + output] * scales.map_or(1., |s| s[input]);
            let mut maximum = 0f32;
            for input in start..end {
                let x = value(input);
                if !x.is_finite() {
                    return Err(invalid("AWQ rescaled weight exceeds finite F32"));
                }
                maximum = maximum.max(x.abs());
            }
            let scale = group_scale(maximum, options.bits)?;
            for base in (start..end).step_by(32) {
                let codes: [i16; 32] =
                    std::array::from_fn(|i| code(value(base + i), scale, options.bits));
                if options.bits == 4 {
                    q4.push(Q4_0Block {
                        scale,
                        values: codes.map(|v| (v + 8) as u8),
                    });
                } else {
                    q8.push(Q8_0Block {
                        scale,
                        values: codes.map(|v| v as i8),
                    });
                }
            }
        }
    }
    if options.bits == 4 {
        Ok(QuantizedData::Q4_0(QuantizedTensor4::from_blocks(
            weight.shape().clone(),
            q4,
        )?))
    } else {
        Ok(QuantizedData::Q8_0(QuantizedTensor::from_blocks(
            weight.shape().clone(),
            q8,
        )?))
    }
}

pub(super) fn group_scale(maximum: f32, bits: usize) -> Result<f32> {
    let max_code = (1_i32 << (bits - 1)) - 1;
    let scale = if maximum == 0. {
        1.
    } else {
        f16::from_f64(f64::from(maximum) / f64::from(max_code)).to_f32()
    };
    if !scale.is_finite() {
        return Err(invalid("AWQ scale exceeds finite F16 storage"));
    }
    Ok(if scale == 0. {
        f16::from_bits(1).to_f32()
    } else {
        scale
    })
}

pub(super) fn code(value: f32, scale: f32, bits: usize) -> i16 {
    let min = -(1_i32 << (bits - 1));
    let max = (1_i32 << (bits - 1)) - 1;
    (f64::from(value) / f64::from(scale))
        .round_ties_even()
        .clamp(f64::from(min), f64::from(max)) as i16
}

pub(super) fn dequantize(data: &QuantizedData, bits: usize) -> Result<Tensor> {
    let (k, n) = matrix_shape(data.shape())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(k * n)
        .map_err(|_| invalid("cannot allocate AWQ decoded matrix"))?;
    values.resize(k * n, 0.);
    for output in 0..n {
        for input in 0..k {
            let block = output * (k / 32) + input / 32;
            let offset = input % 32;
            values[input * n + output] = match (data, bits) {
                (QuantizedData::Q4_0(t), 4) => {
                    t.blocks()[block].scale * (f32::from(t.blocks()[block].values[offset]) - 8.)
                }
                (QuantizedData::Q8_0(t), 8) => {
                    t.blocks()[block].scale * f32::from(t.blocks()[block].values[offset])
                }
                _ => {
                    return Err(invalid(
                        "AWQ provider expects its signed output-row encoding",
                    ))
                }
            };
        }
    }
    Ok(Tensor::from_f32(data.shape().clone(), values)?)
}
