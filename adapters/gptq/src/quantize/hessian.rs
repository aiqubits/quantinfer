use quantinfer_core::Tensor;
use quantinfer_quant::Result;

use super::{invalid, zeroes};

/// Full projection-input Gram statistics, not a diagonal imatrix.
/// Each row of a [tokens, channels] batch is one observation. The lower
/// triangle stores sum(x_i * x_j); normalization uses H = 2 X^T X / rows.
#[derive(Debug, Clone, PartialEq)]
pub struct HessianCalibration {
    pub(super) channels: usize,
    pub(super) rows: usize,
    pub(super) gram: Vec<f64>,
}

impl HessianCalibration {
    /// Bound the statistics allocation separately from quantization workspace.
    pub fn new(channels: usize, maximum_bytes: usize) -> Result<Self> {
        let count = channels
            .checked_mul(channels)
            .ok_or_else(|| invalid("Hessian shape overflows"))?;
        let bytes = count
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| invalid("Hessian size overflows"))?;
        if channels == 0 || bytes > maximum_bytes {
            return Err(invalid(
                "Hessian exceeds the statistics budget or has zero channels",
            ));
        }
        Ok(Self {
            channels,
            rows: 0,
            gram: zeroes(count)?,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Validate the entire batch before updating the statistics.
    pub fn observe_rows(&mut self, input: &Tensor) -> Result<()> {
        let &[rows, channels] = input.shape().dims() else {
            return Err(invalid("calibration requires a [tokens, channels] matrix"));
        };
        if channels != self.channels || rows == 0 {
            return Err(invalid("calibration shape disagrees with Hessian channels"));
        }
        let count = self
            .rows
            .checked_add(rows)
            .ok_or_else(|| invalid("calibration row count overflows"))?;
        let values = input.as_f32()?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(invalid("calibration activations must be finite"));
        }
        for row in values.chunks_exact(channels) {
            for i in 0..channels {
                for j in 0..=i {
                    // Finite F32 products and a usize-bounded observation count
                    // remain within F64 range, even at the F32 extrema.
                    self.gram[i * channels + j] += f64::from(row[i]) * f64::from(row[j]);
                }
            }
        }
        self.rows = count;
        Ok(())
    }

    pub(super) fn value(&self, row: usize, column: usize) -> f64 {
        self.gram[row.max(column) * self.channels + row.min(column)] * (2.0 / self.rows as f64)
    }
}
