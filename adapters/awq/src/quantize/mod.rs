//! Activation-aware scale search with explicit equivalent transformations.
mod clipping;
mod packing;
mod statistics;
mod transforms;

pub use clipping::AwqClippingPlan;
pub use statistics::AwqCalibration;

use quantinfer_core::{
    DType, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor, OUTPUT_ROWS_LAYOUT,
};
use quantinfer_quant::{QuantizationError, QuantizationProvider, QuantizedData, Result};

#[derive(Debug, Clone, Copy)]
pub struct AwqOptions {
    pub bits: usize,
    pub group_size: usize,
    /// Ratios 0/grid_points through (grid_points-1)/grid_points.
    pub grid_points: usize,
    /// Algorithm-owned candidates; excludes caller-owned model/activations
    /// and allocations made by the caller's block evaluator.
    pub maximum_workspace_bytes: usize,
}
impl Default for AwqOptions {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            grid_points: 20,
            maximum_workspace_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AwqScalePlan {
    scales: Vec<f32>,
    pub ratio: f64,
    pub loss: f64,
    pub candidate_losses: Vec<f64>,
}
impl AwqScalePlan {
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Fold x -> x/s into RMSNorm/LayerNorm's learned multiplicative weight.
    /// Bias-free dense models are the current production profile.
    pub fn normalization(&self, weight: &Tensor) -> Result<Tensor> {
        transforms::normalization(weight, &self.scales)
    }

    /// Multiply input-channel rows of a logical [input,output] matrix by s.
    pub fn input_weight(&self, weight: &Tensor) -> Result<Tensor> {
        transforms::input_weight(weight, &self.scales)
    }

    /// Divide output-channel columns by s. Together with input_weight on a
    /// following linear map this preserves the bias-free linear function.
    /// In SwiGLU, apply it to the up projection, not the nonlinear gate.
    pub fn output_weight(&self, weight: &Tensor) -> Result<Tensor> {
        transforms::output_weight(weight, &self.scales)
    }
}

#[derive(Debug, Clone, Default)]
pub struct AwqProvider {
    options: AwqOptions,
}
impl AwqProvider {
    pub fn new(options: AwqOptions) -> Result<Self> {
        if !matches!(options.bits, 4 | 8)
            || options.group_size == 0
            || !options.group_size.is_multiple_of(32)
            || !(2..=100).contains(&options.grid_points)
            || options.maximum_workspace_bytes == 0
        {
            return Err(invalid("native AWQ needs bits 4/8, group multiple of 32, 2..=100 grid points and positive workspace"));
        }
        Ok(Self { options })
    }
    pub fn options(&self) -> &AwqOptions {
        &self.options
    }

    /// Search actual block reconstruction loss, not weight error or a diagonal
    /// proxy. The evaluator receives effective [input,output] floating matrices
    /// Q(W*s)/s in the same order as `weights`. It must evaluate the reference
    /// attention/FFN/linear block on its fixed calibration inputs.
    ///
    /// This callback allows architecture-independent search. Production must
    /// fold the selected transform and quantize the transformed floating model;
    /// it must not quantize the simulated matrices returned to the evaluator.
    pub fn search_scales(
        &self,
        weights: &[&Tensor],
        calibration: &AwqCalibration,
        mut evaluate: impl FnMut(Vec<Tensor>) -> Result<f64>,
    ) -> Result<AwqScalePlan> {
        if weights.is_empty() || weights.len() > 8 || calibration.rows() == 0 {
            return Err(invalid(
                "AWQ requires observed calibration and 1..=8 related projections",
            ));
        }
        let mut elements = 0usize;
        for weight in weights {
            self.validate(weight)?;
            if weight.shape().dims()[0] != calibration.channels() {
                return Err(invalid("AWQ calibration channels disagree with projection"));
            }
            elements = elements
                .checked_add(weight.shape().num_elements()?)
                .ok_or_else(|| invalid("AWQ group size overflows"))?;
        }
        let workspace = elements
            .checked_mul(12)
            .and_then(|n| {
                calibration
                    .channels()
                    .checked_mul(24)
                    .and_then(|c| n.checked_add(c))
            })
            .and_then(|n| n.checked_add(weights.len() * 128 + self.options.grid_points * 8))
            .ok_or_else(|| invalid("AWQ search workspace overflows"))?;
        if workspace > self.options.maximum_workspace_bytes {
            return Err(invalid("AWQ search exceeds workspace budget"));
        }
        let means = calibration.mean_absolute()?;
        let mut losses = buffer(self.options.grid_points)?;
        let mut best = None;
        for index in 0..self.options.grid_points {
            let ratio = index as f64 / self.options.grid_points as f64;
            let mut values = buffer(means.len())?;
            values.extend(means.iter().map(|v| v.powf(ratio).max(1e-4)));
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(0., f64::max);
            let normalization = (min * max).sqrt();
            let mut scales = buffer(values.len())?;
            scales.extend(values.drain(..).map(|v| (v / normalization) as f32));
            transforms::validate_scales(&scales, calibration.channels())?;
            let mut simulated = buffer(weights.len())?;
            for weight in weights {
                let packed = packing::quantize(weight, Some(&scales), self.options)?;
                let restored = packing::dequantize(&packed, self.options.bits)?;
                simulated.push(transforms::undo_input_weight(&restored, &scales)?);
            }
            let loss = evaluate(simulated)?;
            if !loss.is_finite() || loss < 0. {
                return Err(invalid("AWQ block evaluator returned invalid loss"));
            }
            losses.push(loss);
            if best
                .as_ref()
                .is_none_or(|plan: &AwqScalePlan| loss < plan.loss)
            {
                best = Some(AwqScalePlan {
                    scales,
                    ratio,
                    loss,
                    candidate_losses: Vec::new(),
                });
            }
        }
        let mut best = best.ok_or_else(|| invalid("AWQ search produced no candidate"))?;
        best.candidate_losses = losses;
        Ok(best)
    }

    /// Final production after equivalent transforms and optional clipping.
    /// The input is floating, with all coupled scaling transforms already
    /// applied. The output keeps signed codes/F16 scales in output-row order.
    pub fn quantize_transformed(&self, weight: &Tensor) -> Result<QuantizedData> {
        self.validate(weight)?;
        let count = weight.shape().num_elements()?;
        if count
            .checked_mul(4)
            .is_none_or(|n| n > self.options.maximum_workspace_bytes)
        {
            return Err(invalid("AWQ production exceeds workspace budget"));
        }
        packing::quantize(weight, None, self.options)
    }
}

impl QuantizationProvider for AwqProvider {
    fn scheme(&self) -> QuantizationScheme {
        if self.options.bits == 4 {
            QuantizationScheme::Q4_0
        } else {
            QuantizationScheme::Q8_0
        }
    }
    fn storage_descriptor(&self, shape: Shape) -> Result<TensorStorageDescriptor> {
        matrix_shape(&shape)?;
        let mut descriptor = TensorStorageDescriptor::quantized(
            if self.options.bits == 4 {
                DType::Q4_0
            } else {
                DType::Q8_0
            },
            self.scheme(),
            shape,
            32,
        )?;
        descriptor.physical_layout = OUTPUT_ROWS_LAYOUT.into();
        Ok(descriptor)
    }
    fn validate(&self, input: &Tensor) -> Result<()> {
        matrix_shape(input.shape())?;
        if input.as_f32()?.iter().any(|x| !x.is_finite()) {
            return Err(invalid("AWQ weights must be finite F32"));
        }
        Ok(())
    }
    fn quantize(&self, _input: &Tensor) -> Result<QuantizedData> {
        Err(invalid("AWQ requires activation search and equivalent transforms; use search_scales then quantize_transformed"))
    }
    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        packing::dequantize(input, self.options.bits)
    }
    fn export_layout(&self) -> &'static str {
        OUTPUT_ROWS_LAYOUT
    }
}

fn matrix_shape(shape: &Shape) -> Result<(usize, usize)> {
    match shape.dims() {
        &[k, n] if k > 0 && n > 0 && k.is_multiple_of(32) && k.checked_mul(n).is_some() => {
            Ok((k, n))
        }
        _ => Err(invalid(
            "native AWQ needs [input,output] with positive input divisible by 32",
        )),
    }
}
fn invalid(message: impl Into<String>) -> QuantizationError {
    QuantizationError::InvalidInput(message.into())
}
fn buffer<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("cannot allocate AWQ workspace"))?;
    Ok(values)
}

#[cfg(test)]
mod tests;
