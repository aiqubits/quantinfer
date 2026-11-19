use super::{
    buffer, invalid, matrix_shape, AutoRoundCalibration, AutoRoundOptions, AutoRoundProvider,
    AutoRoundResult,
};
use half::f16;
use quantinfer_core::{Q4_0Block, Q8_0Block, QuantizedTensor, QuantizedTensor4, Tensor};
use quantinfer_quant::{QuantizedData, Result};
use rayon::prelude::*;

pub(super) fn optimize(
    provider: &AutoRoundProvider,
    weight: &Tensor,
    calibration: &AutoRoundCalibration,
) -> Result<AutoRoundResult> {
    let options = provider.options;
    let (k, n) = matrix_shape(weight.shape())?;
    if calibration.channels() != k
        || calibration.rows() == 0
        || calibration.values().iter().all(|&v| v == 0.)
    {
        return Err(invalid(
            "AutoRound requires matching nonzero calibration activations",
        ));
    }
    let groups = k.div_ceil(options.group_size);
    let per_worker = k
        .checked_mul(56)
        .and_then(|n| n.checked_add(groups.checked_mul(48)?))
        .and_then(|n| n.checked_add(calibration.rows().checked_mul(8)?))
        .ok_or_else(|| invalid("AutoRound worker workspace overflows"))?;
    let required = n
        .checked_mul(k)
        .and_then(|n| n.checked_mul(4))
        .and_then(|result| {
            result.checked_add(n.checked_mul(groups.checked_mul(4)?.checked_add(96)?)?)
        })
        .and_then(|result| result.checked_add(per_worker.checked_mul(options.threads.min(n))?))
        .ok_or_else(|| invalid("AutoRound workspace overflows"))?;
    if required > options.maximum_workspace_bytes {
        return Err(invalid("AutoRound search exceeds workspace budget"));
    }
    let source = weight.as_f32()?;
    let rows: Vec<Row> = provider.pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|o| optimize_row(source, n, o, calibration, options))
            .collect::<Result<Vec<_>>>()
    })?;
    let (initial_mse, best_mse, improved_outputs) =
        rows.iter()
            .fold((0., 0., 0), |(initial, best, improved), row| {
                (
                    initial + row.initial,
                    best + row.loss,
                    improved + usize::from(row.loss < row.initial),
                )
            });
    let quantized = if options.bits == 4 {
        let mut blocks = reserve(n * (k / 32))?;
        for row in rows {
            for (block, chunk) in row.codes.as_chunks::<32>().0.iter().enumerate() {
                blocks.push(Q4_0Block {
                    scale: row.scales[block * 32 / options.group_size],
                    values: chunk.map(|c| (i16::from(c) + 8) as u8),
                });
            }
        }
        QuantizedData::Q4_0(QuantizedTensor4::from_blocks(
            weight.shape().clone(),
            blocks,
        )?)
    } else {
        let mut blocks = reserve(n * (k / 32))?;
        for row in rows {
            for (block, chunk) in row.codes.as_chunks::<32>().0.iter().enumerate() {
                blocks.push(Q8_0Block {
                    scale: row.scales[block * 32 / options.group_size],
                    values: *chunk,
                });
            }
        }
        QuantizedData::Q8_0(QuantizedTensor::from_blocks(
            weight.shape().clone(),
            blocks,
        )?)
    };
    Ok(AutoRoundResult {
        weights: quantized,
        initial_mse: initial_mse / n as f64,
        best_mse: best_mse / n as f64,
        improved_outputs,
    })
}

struct Row {
    codes: Vec<i8>,
    scales: Vec<f32>,
    initial: f64,
    loss: f64,
}

fn optimize_row(
    source: &[f32],
    outputs: usize,
    output: usize,
    calibration: &AutoRoundCalibration,
    options: AutoRoundOptions,
) -> Result<Row> {
    let k = calibration.channels();
    let group = options.group_size;
    let max_code = ((1i16 << (options.bits - 1)) - 1) as i8;
    let min_code = -max_code - 1;
    let groups = k.div_ceil(group);
    let mut weights = reserve(k)?;
    weights.extend(
        source
            .chunks_exact(outputs)
            .map(|row| f64::from(row[output])),
    );
    let mut base_scales = reserve(groups)?;
    for weights in weights.chunks(group) {
        let peak = weights.iter().map(|w| w.abs()).fold(0., f64::max);
        base_scales.push(if peak == 0. {
            1.
        } else {
            peak / f64::from(max_code)
        });
    }
    let mut factors = buffer(groups, 1.)?;
    let mut h = buffer(k, 0.0)?;
    let mut step = buffer(groups, 0.0f32)?;
    let mut codes = buffer(k, 0i8)?;
    let mut residual_weights = buffer(k, 0.0)?;
    let mut ste_active = buffer(k, false)?;
    let mut output_residual = buffer(calibration.rows(), 0.0)?;
    let mut gradients = buffer(k, 0.0)?;
    let mut best = Row {
        codes: buffer(k, 0)?,
        scales: buffer(groups, 0.)?,
        initial: 0.,
        loss: f64::INFINITY,
    };
    let initial_lr = 1. / options.iterations as f64;
    let round_lr = options.rounding_lr.unwrap_or(initial_lr);
    let clip_lr = options.clipping_lr.unwrap_or(initial_lr);
    for iteration in 0..=options.iterations {
        for (i, scale) in step.iter_mut().enumerate() {
            *scale = f16::from_f64(base_scales[i] * factors[i])
                .to_f32()
                .max(f16::from_bits(1).to_f32());
            if !scale.is_finite() {
                return Err(invalid("AutoRound scale is outside the finite F16 grid"));
            }
        }
        for (i, &weight) in weights.iter().enumerate() {
            let scale = f64::from(step[i / group]);
            let rounded = (weight / scale + h[i]).round_ties_even();
            ste_active[i] = rounded >= f64::from(min_code) && rounded <= f64::from(max_code);
            codes[i] = rounded.clamp(f64::from(min_code), f64::from(max_code)) as i8;
            residual_weights[i] = f64::from(f32::from(codes[i]) * step[i / group]) - weight;
        }
        let mut squared = 0.0;
        for (residual, x) in output_residual
            .iter_mut()
            .zip(calibration.values().chunks_exact(k))
        {
            *residual = dot(&residual_weights, x);
            squared += *residual * *residual;
        }
        let loss = squared / calibration.rows() as f64;
        if !loss.is_finite() {
            return Err(invalid("non-finite AutoRound reconstruction error"));
        }
        if iteration == 0 {
            best.initial = loss;
        }
        if loss < best.loss {
            best.loss = loss;
            best.codes.copy_from_slice(&codes);
            best.scales.copy_from_slice(&step);
        }
        if iteration == options.iterations {
            break;
        }
        gradients.fill(0.);
        for (&error, input) in output_residual
            .iter()
            .zip(calibration.values().chunks_exact(k))
        {
            for (grad, &value) in gradients.iter_mut().zip(input) {
                *grad += error * f64::from(value);
            }
        }
        let decay = 1. - iteration as f64 / options.iterations as f64;
        for (g, factor) in factors.iter_mut().enumerate() {
            let start = g * group;
            let end = start + group.min(k - start);
            let mut clipping_gradient = 0.;
            let scale = f64::from(step[g]);
            for i in start..end {
                // Round and F16 conversion use straight-through derivatives.
                // Saturated integer codes have zero derivative through clamp.
                let code = f64::from(codes[i]);
                let derivative = if ste_active[i] {
                    h[i] = (h[i] - round_lr * decay * sign(gradients[i])).clamp(-0.5, 0.5);
                    code - weights[i] / scale
                } else {
                    code
                };
                clipping_gradient += derivative * gradients[i];
            }
            *factor = (*factor - clip_lr * decay * sign(clipping_gradient)).clamp(0., 1.);
        }
    }
    Ok(best)
}

fn sign(value: f64) -> f64 {
    if value > 0. {
        1.
    } else if value < 0. {
        -1.
    } else {
        0.
    }
}

fn dot(weights: &[f64], activations: &[f32]) -> f64 {
    let mut sums = [0.; 8];
    for (w, x) in weights
        .as_chunks::<8>()
        .0
        .iter()
        .zip(activations.as_chunks::<8>().0)
    {
        for i in 0..8 {
            sums[i] += w[i] * f64::from(x[i]);
        }
    }
    sums.into_iter().sum()
}

fn reserve<T>(len: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| invalid("cannot allocate AutoRound weights"))?;
    Ok(result)
}

pub(super) fn dequantize(data: &QuantizedData, bits: usize) -> Result<Tensor> {
    let (k, n) = matrix_shape(data.shape())?;
    let mut values = buffer(k * n, 0.)?;
    for output in 0..n {
        for input in 0..k {
            let block = output * (k / 32) + input / 32;
            let offset = input % 32;
            values[input * n + output] = match (data, bits) {
                (QuantizedData::Q4_0(tensor), 4) => {
                    let block = &tensor.blocks()[block];
                    block.scale * (f32::from(block.values[offset]) - 8.)
                }
                (QuantizedData::Q8_0(tensor), 8) => {
                    let block = &tensor.blocks()[block];
                    block.scale * f32::from(block.values[offset])
                }
                _ => return Err(invalid("AutoRound storage encoding disagrees")),
            };
        }
    }
    Ok(Tensor::from_f32(data.shape().clone(), values)?)
}
