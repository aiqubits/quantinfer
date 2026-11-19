//! Per-output/group AWQ clipping on actual activation correlations.

use super::{buffer, invalid, matrix_shape, packing, AwqProvider};
use quantinfer_core::Tensor;
use quantinfer_quant::{QuantizationProvider, Result};

#[derive(Debug, Clone)]
pub struct AwqClippingPlan {
    /// Floating transformed weights. Quantize these exactly once at publication.
    pub weights: Tensor,
    /// Bounds in [output, input_group] order; the final group may be shorter.
    pub bounds: Vec<f32>,
    pub group_size: usize,
    pub calibration_rows: usize,
    /// Sum of per-output/group mean squared reconstruction errors. This omits
    /// cross-group terms and is not the complete projection or model loss.
    pub group_loss_before: f64,
    pub group_loss_after: f64,
}

impl AwqProvider {
    /// Search the ten shrink factors 1, .95, ..., .55 used by AWQ's clipping
    /// profile, with this provider's signed F16-scale quantizer. The caller
    /// supplies activations after folding all equivalent scaling transforms.
    /// Model orchestration must skip Q/K clipping and validate complete blocks.
    pub fn search_clipping(
        &self,
        weight: &Tensor,
        activations: &Tensor,
    ) -> Result<AwqClippingPlan> {
        self.validate(weight)?;
        let (k, n) = matrix_shape(weight.shape())?;
        let &[rows, channels] = activations.shape().dims() else {
            return Err(invalid("AWQ clipping needs [token_rows,input] activations"));
        };
        if rows == 0 || channels != k {
            return Err(invalid("AWQ clipping activation shape is inconsistent"));
        }
        let x = activations.as_f32()?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err(invalid("AWQ clipping activations must be finite"));
        }
        let size = self.options.group_size.min(k);
        let groups = k.div_ceil(self.options.group_size);
        let bounds_count = n
            .checked_mul(groups)
            .ok_or_else(|| invalid("AWQ clipping bounds overflow"))?;
        let workspace = size
            .checked_mul(size)
            .and_then(|m| m.checked_mul(8))
            .and_then(|m| size.checked_mul(8).and_then(|e| m.checked_add(e)))
            .and_then(|m| {
                weight
                    .shape()
                    .num_elements()
                    .ok()?
                    .checked_mul(4)?
                    .checked_add(m)
            })
            .and_then(|m| bounds_count.checked_mul(4)?.checked_add(m))
            .ok_or_else(|| invalid("AWQ clipping workspace overflows"))?;
        if workspace > self.options.maximum_workspace_bytes {
            return Err(invalid("AWQ clipping exceeds workspace budget"));
        }
        let source = weight.as_f32()?;
        let mut clipped = buffer(source.len())?;
        clipped.extend_from_slice(source);
        let mut bounds = buffer(bounds_count)?;
        bounds.resize(bounds_count, 0.);
        let mut gram = buffer(size * size)?;
        let mut errors = buffer(size)?;
        let (mut before, mut after) = (0., 0.);
        for (group, start) in (0..k).step_by(self.options.group_size).enumerate() {
            let end = start + self.options.group_size.min(k - start);
            let width = end - start;
            gram.clear();
            gram.resize(width * width, 0f64);
            for row in x.chunks_exact(k) {
                for i in 0..width {
                    for j in 0..=i {
                        gram[i * width + j] +=
                            f64::from(row[start + i]) * f64::from(row[start + j]) / rows as f64;
                    }
                }
            }
            for output in 0..n {
                let maximum = (start..end)
                    .map(|i| source[i * n + output].abs())
                    .fold(0., f32::max);
                let (mut best_loss, mut best_bound) = (f64::INFINITY, maximum);
                for candidate in 0..10 {
                    let bound = maximum * (1. - candidate as f32 / 20.);
                    let scale = packing::group_scale(bound, self.options.bits)?;
                    errors.clear();
                    for input in start..end {
                        let original = source[input * n + output];
                        let code =
                            packing::code(original.clamp(-bound, bound), scale, self.options.bits);
                        let decoded = scale * f32::from(code);
                        errors.push(f64::from(decoded) - f64::from(original));
                    }
                    let loss = quadratic(&errors, &gram)?;
                    if candidate == 0 {
                        before += loss;
                    }
                    if loss < best_loss {
                        best_loss = loss;
                        best_bound = bound;
                    }
                }
                after += best_loss;
                bounds[output * groups + group] = best_bound;
                for input in start..end {
                    clipped[input * n + output] =
                        source[input * n + output].clamp(-best_bound, best_bound);
                }
            }
        }
        if !before.is_finite() || !after.is_finite() {
            return Err(invalid("AWQ clipping loss overflowed"));
        }
        Ok(AwqClippingPlan {
            weights: Tensor::from_f32(weight.shape().clone(), clipped)?,
            bounds,
            group_size: self.options.group_size,
            calibration_rows: rows,
            group_loss_before: before,
            group_loss_after: after,
        })
    }
}

fn quadratic(error: &[f64], lower_gram: &[f64]) -> Result<f64> {
    let mut sum = 0.;
    let mut magnitude = 0.;
    for i in 0..error.len() {
        for j in 0..=i {
            let term = error[i]
                * error[j]
                * lower_gram[i * error.len() + j]
                * if i == j { 1. } else { 2. };
            sum += term;
            magnitude += term.abs();
        }
    }
    // A Gram quadratic is nonnegative. Permit only floating summation noise
    // from cancellation, not a materially negative or non-finite loss.
    if !sum.is_finite() || sum < -64. * f64::EPSILON * magnitude {
        return Err(invalid("AWQ clipping covariance produced invalid loss"));
    }
    Ok(sum.max(0.))
}

#[cfg(test)]
mod tests;
