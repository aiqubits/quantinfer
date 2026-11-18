//! Quantization provider contracts and native provider implementations.

use quantinfer_core::{
    QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K,
    QuantizedTensorQ6K, Tensor, TensorStorageDescriptor,
};

use crate::schemes::validate_finite;
use crate::{QuantizationError, Result};

#[derive(Debug, Clone, PartialEq)]
#[allow(non_camel_case_types)]
pub enum QuantizedData {
    Float(Tensor),
    Q8_0(QuantizedTensor),
    Q4_0(QuantizedTensor4),
    Q4_K(QuantizedTensorQ4K),
    Q5_K(QuantizedTensorQ5K),
    Q6_K(QuantizedTensorQ6K),
}

impl QuantizedData {
    pub fn scheme(&self) -> QuantizationScheme {
        match self {
            Self::Float(_) => QuantizationScheme::None,
            Self::Q8_0(_) => QuantizationScheme::Q8_0,
            Self::Q4_0(_) => QuantizationScheme::Q4_0,
            Self::Q4_K(_) => QuantizationScheme::Q4_K,
            Self::Q5_K(_) => QuantizationScheme::Q5_K,
            Self::Q6_K(_) => QuantizationScheme::Q6_K,
        }
    }

    pub fn shape(&self) -> &quantinfer_core::Shape {
        match self {
            Self::Float(tensor) => tensor.shape(),
            Self::Q8_0(tensor) => tensor.shape(),
            Self::Q4_0(tensor) => tensor.shape(),
            Self::Q4_K(tensor) => tensor.shape(),
            Self::Q5_K(tensor) => tensor.shape(),
            Self::Q6_K(tensor) => tensor.shape(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationStats {
    pub elements: usize,
    pub min: f32,
    pub max: f32,
    pub mean_abs: f32,
    pub max_abs: f32,
}

impl QuantizationStats {
    pub fn inspect(input: &Tensor) -> Result<Self> {
        let values = input.as_f32()?;
        if values.is_empty() {
            return Err(QuantizationError::InvalidInput(
                "cannot inspect an empty tensor".to_owned(),
            ));
        }
        validate_finite(values)?;
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_abs = values
            .iter()
            .map(|value| f64::from(*value).abs())
            .sum::<f64>();
        Ok(Self {
            elements: values.len(),
            min,
            max,
            mean_abs: (sum_abs / values.len() as f64) as f32,
            max_abs: min.abs().max(max.abs()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationErrorReport {
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    /// `||reference - restored||₂ / ||reference||₂`, with F64 accumulation.
    /// Two zero tensors have zero error; a zero reference with nonzero error
    /// or a metric outside finite F32 range makes `error_report` fail.
    pub relative_l2_error: f32,
}

pub trait QuantizationProvider {
    fn scheme(&self) -> QuantizationScheme;
    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor>;
    fn inspect(&self, input: &Tensor) -> Result<QuantizationStats> {
        QuantizationStats::inspect(input)
    }
    fn validate(&self, input: &Tensor) -> Result<()>;
    fn quantize(&self, input: &Tensor) -> Result<QuantizedData>;
    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor>;
    fn error_report(
        &self,
        reference: &Tensor,
        input: &QuantizedData,
    ) -> Result<QuantizationErrorReport> {
        let restored = self.dequantize(input)?;
        let reference_values = reference.as_f32()?;
        let restored_values = restored.as_f32()?;
        if reference.shape() != restored.shape() {
            return Err(QuantizationError::InvalidInput(
                "error report tensors must have the same shape".to_owned(),
            ));
        }
        if reference_values.len() != restored_values.len() {
            return Err(QuantizationError::InvalidInput(
                "error report tensors must have the same number of elements".to_owned(),
            ));
        }
        validate_finite(reference_values)?;
        validate_finite(restored_values)?;
        let mut max_absolute_error: f64 = 0.0;
        let mut sum_absolute_error: f64 = 0.0;
        let mut sum_squared_reference: f64 = 0.0;
        let mut sum_squared_error: f64 = 0.0;
        for (reference, restored) in reference_values.iter().zip(restored_values) {
            let reference = f64::from(*reference);
            let error = reference - f64::from(*restored);
            max_absolute_error = max_absolute_error.max(error.abs());
            sum_absolute_error += error.abs();
            sum_squared_reference += reference * reference;
            sum_squared_error += error * error;
        }
        let relative_l2_error = if sum_squared_reference == 0.0 {
            if sum_squared_error != 0.0 {
                return Err(QuantizationError::InvalidInput(
                    "relative L2 error is undefined for a zero reference norm".to_owned(),
                ));
            }
            0.0
        } else {
            sum_squared_error.sqrt() / sum_squared_reference.sqrt()
        };
        let report = QuantizationErrorReport {
            max_absolute_error: max_absolute_error as f32,
            mean_absolute_error: (sum_absolute_error / reference_values.len().max(1) as f64) as f32,
            relative_l2_error: relative_l2_error as f32,
        };
        if !report.max_absolute_error.is_finite()
            || !report.mean_absolute_error.is_finite()
            || !report.relative_l2_error.is_finite()
        {
            return Err(QuantizationError::InvalidInput(
                "quantization error metrics exceed finite F32 range".to_owned(),
            ));
        }
        Ok(report)
    }
    /// Measure the reconstruction after rounding the current block encodings'
    /// scales to their F16 storage precision. This does not verify a file.
    fn error_report_for_storage(
        &self,
        reference: &Tensor,
        input: &QuantizedData,
    ) -> Result<QuantizationErrorReport> {
        self.error_report(reference, &input.with_storage_precision()?)
    }
    fn export_layout(&self) -> &'static str;
}

mod float;
mod q4;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8;
mod storage;

#[cfg(test)]
mod tests;

pub use float::FloatProvider;
pub use q4::NativeQ4_0Provider;
pub use q4_k::NativeQ4KProvider;
pub use q5_k::NativeQ5KProvider;
pub use q6_k::NativeQ6KProvider;
pub use q8::{GgmlQ8_0Provider, NativeQ8_0Provider};
