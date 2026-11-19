//! Full-Hessian GPTQ with static signed groups and lossless native packing.

mod hessian;
mod linalg;
mod packing;

#[cfg(test)]
mod tests;

pub use hessian::HessianCalibration;

use quantinfer_core::{
    DType, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor, OUTPUT_ROWS_LAYOUT,
};
use quantinfer_quant::{
    CalibratedQuantizationProvider, QuantizationError, QuantizationProvider, QuantizedData, Result,
};

#[derive(Debug, Clone, Copy)]
pub struct GptqQuantizationOptions {
    /// Four/eight bits map exactly to native Q4_0/Q8_0 storage. Import APIs
    /// separately retain their support for external two-bit checkpoints.
    pub bits: usize,
    /// Static contiguous input groups, a positive multiple of 32.
    pub group_size: usize,
    /// Positive fraction of mean Hessian diagonal, added before inversion.
    pub damping: f64,
    /// Process high-activation channels first while preserving static groups.
    pub activation_order: bool,
    /// Conservative bound for algorithm-owned buffers; excludes caller-owned
    /// weights and calibration statistics. This is not a process RSS limit.
    pub maximum_workspace_bytes: usize,
}

impl Default for GptqQuantizationOptions {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            damping: 0.01,
            activation_order: false,
            maximum_workspace_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GptqProvider {
    options: GptqQuantizationOptions,
}

impl GptqProvider {
    pub fn new(options: GptqQuantizationOptions) -> Result<Self> {
        if !matches!(options.bits, 4 | 8)
            || options.group_size == 0
            || !options.group_size.is_multiple_of(32)
            || !options.damping.is_finite()
            || options.damping <= 0.0
            || options.maximum_workspace_bytes == 0
        {
            return Err(invalid("invalid native GPTQ options (bits 4/8, group multiple of 32, positive damping/budget required)"));
        }
        Ok(Self { options })
    }

    pub fn options(&self) -> &GptqQuantizationOptions {
        &self.options
    }
}

impl QuantizationProvider for GptqProvider {
    fn scheme(&self) -> QuantizationScheme {
        if self.options.bits == 4 {
            QuantizationScheme::Q4_0
        } else {
            QuantizationScheme::Q8_0
        }
    }

    fn storage_descriptor(&self, shape: Shape) -> Result<TensorStorageDescriptor> {
        matrix_shape(&shape)?;
        let dtype = if self.options.bits == 4 {
            DType::Q4_0
        } else {
            DType::Q8_0
        };
        let mut descriptor = TensorStorageDescriptor::quantized(dtype, self.scheme(), shape, 32)?;
        descriptor.physical_layout = OUTPUT_ROWS_LAYOUT.into();
        Ok(descriptor)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        matrix_shape(input.shape())?;
        if input.as_f32()?.iter().any(|x| !x.is_finite()) {
            return Err(invalid("GPTQ weights must be finite F32"));
        }
        Ok(())
    }

    fn quantize(&self, _input: &Tensor) -> Result<QuantizedData> {
        Err(invalid(
            "GPTQ requires full activation calibration; use quantize_calibrated",
        ))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        packing::dequantize(input, self.options.bits)
    }

    fn export_layout(&self) -> &'static str {
        OUTPUT_ROWS_LAYOUT
    }
}

impl CalibratedQuantizationProvider for GptqProvider {
    type Calibration = HessianCalibration;

    fn quantize_calibrated(
        &self,
        input: &Tensor,
        calibration: &HessianCalibration,
    ) -> Result<QuantizedData> {
        self.validate(input)?;
        let (channels, outputs) = matrix_shape(input.shape())?;
        if calibration.channels != channels || calibration.rows == 0 {
            return Err(invalid(
                "GPTQ requires observed calibration with matching input channels",
            ));
        }
        // Two dense F64 Hessian buffers, F64 working weights, codes/scales and
        // native result blocks, plus channel order and dead-channel flags.
        let workspace = channels
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| {
                channels
                    .checked_mul(outputs)
                    .and_then(|w| w.checked_mul(11))
                    .and_then(|w| n.checked_add(w))
            })
            .and_then(|n| {
                channels
                    .checked_mul(size_of::<usize>() + 1)
                    .and_then(|p| n.checked_add(p))
            })
            .ok_or_else(|| invalid("GPTQ workspace size overflows"))?;
        if workspace > self.options.maximum_workspace_bytes {
            return Err(invalid(format!(
                "GPTQ needs {workspace} workspace bytes, above configured budget"
            )));
        }
        let mut order: Vec<usize> = (0..channels).collect();
        let dead: Vec<bool> = (0..channels)
            .map(|i| calibration.value(i, i) == 0.0)
            .collect();
        if dead.iter().all(|value| *value) {
            return Err(invalid("GPTQ calibration contains no nonzero activations"));
        }
        let source = input.as_f32()?;
        // Reject an unrepresentable native grid before the cubic factorization.
        let scales = packing::scales(source, channels, outputs, self.options, &dead)?;
        if self.options.activation_order {
            order.sort_by(|&a, &b| {
                calibration
                    .value(b, b)
                    .total_cmp(&calibration.value(a, a))
                    .then(a.cmp(&b))
            });
        }
        let mut hessian = zeroes(channels * channels)?;
        for (i, &a) in order.iter().enumerate() {
            for (j, &b) in order.iter().enumerate().take(i + 1) {
                hessian[i * channels + j] = if a == b && dead[a] {
                    1.0
                } else {
                    calibration.value(a, b)
                };
            }
        }
        let mean_diagonal = (0..channels)
            .map(|i| hessian[i * channels + i])
            .sum::<f64>()
            / channels as f64;
        let damp = self.options.damping * mean_diagonal;
        if !damp.is_finite() || damp <= 0.0 {
            return Err(invalid(
                "Hessian damping must remain finite and positive after scaling",
            ));
        }
        for i in 0..channels {
            hessian[i * channels + i] += damp;
        }
        let factor = linalg::inverse_factor(hessian, channels)?;
        let mut working = zeroes(channels * outputs)?;
        for output in 0..outputs {
            for (i, &original) in order.iter().enumerate() {
                working[output * channels + i] = if dead[original] {
                    0.0
                } else {
                    f64::from(source[original * outputs + output])
                };
            }
        }
        let mut codes = Vec::new();
        codes
            .try_reserve_exact(working.len())
            .map_err(|_| invalid("cannot allocate GPTQ codes"))?;
        codes.resize(working.len(), 0_u8);
        let zero = 1_i32 << (self.options.bits - 1);
        let maximum = (1_i32 << self.options.bits) - 1;
        for (i, &original) in order.iter().enumerate() {
            let diagonal = factor[i * channels + i];
            for output in 0..outputs {
                let scale = f64::from(scales[output * (channels / 32) + original / 32]);
                let weight = working[output * channels + i];
                let code = ((weight / scale).round_ties_even() + f64::from(zero))
                    .clamp(0.0, f64::from(maximum)) as u8;
                codes[output * channels + original] = code;
                let restored = scale * (f64::from(code) - f64::from(zero));
                let error = (weight - restored) / diagonal;
                if !error.is_finite() {
                    return Err(invalid("GPTQ feedback exceeds finite F64 range"));
                }
                for j in i + 1..channels {
                    let next = &mut working[output * channels + j];
                    *next -= error * factor[i * channels + j];
                    if !next.is_finite() {
                        return Err(invalid("GPTQ feedback weight exceeds finite F64 range"));
                    }
                }
            }
        }
        packing::pack(input.shape(), &codes, &scales, self.options.bits)
    }
}

fn matrix_shape(shape: &Shape) -> Result<(usize, usize)> {
    match shape.dims() {
        &[channels, outputs] if channels > 0 && outputs > 0 && channels.is_multiple_of(32) => {
            Ok((channels, outputs))
        }
        _ => Err(invalid(
            "native GPTQ requires weight shape [input, output] and input divisible by 32",
        )),
    }
}

fn invalid(message: impl Into<String>) -> QuantizationError {
    QuantizationError::InvalidInput(message.into())
}

fn zeroes(count: usize) -> Result<Vec<f64>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| invalid("cannot allocate GPTQ workspace"))?;
    result.resize(count, 0.0);
    Ok(result)
}
