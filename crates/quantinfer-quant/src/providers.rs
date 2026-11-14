//! Quantization provider contracts and native provider implementations.

use quantinfer_core::{
    DType, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor, TensorStorageDescriptor, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};

use crate::schemes::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0, validate_finite,
};
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
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_abs = values.iter().map(|value| value.abs()).sum::<f32>();
        Ok(Self {
            elements: values.len(),
            min,
            max,
            mean_abs: sum_abs / values.len() as f32,
            max_abs: min.abs().max(max.abs()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationErrorReport {
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
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
        if reference_values.len() != restored_values.len() {
            return Err(QuantizationError::InvalidInput(
                "error report tensors must have the same number of elements".to_owned(),
            ));
        }
        let mut max_absolute_error: f32 = 0.0;
        let mut sum_absolute_error: f32 = 0.0;
        let mut sum_squared_reference: f32 = 0.0;
        let mut sum_squared_error: f32 = 0.0;
        for (reference, restored) in reference_values.iter().zip(restored_values) {
            let error = *reference - *restored;
            max_absolute_error = max_absolute_error.max(error.abs());
            sum_absolute_error += error.abs();
            sum_squared_reference += reference * reference;
            sum_squared_error += error * error;
        }
        let denominator = sum_squared_reference.sqrt().max(f32::EPSILON);
        Ok(QuantizationErrorReport {
            max_absolute_error,
            mean_absolute_error: sum_absolute_error / reference_values.len().max(1) as f32,
            relative_l2_error: sum_squared_error.sqrt() / denominator,
        })
    }
    fn export_layout(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FloatProvider;

impl QuantizationProvider for FloatProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::None
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::floating(DType::F32, shape)?)
    }

    fn validate(&self, _input: &Tensor) -> Result<()> {
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Float(input.clone()))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Float(tensor) => Ok(tensor.clone()),
            _ => Err(QuantizationError::InvalidInput(
                "FloatProvider cannot dequantize a quantized tensor".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "f32"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ8_0Provider;

impl QuantizationProvider for NativeQ8_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q8_0
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q8_0,
            QuantizationScheme::Q8_0,
            shape,
            Q8_0_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        let values = input.as_f32()?;
        validate_finite(values)?;
        if !values.len().is_multiple_of(Q8_0_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q8_0 requires a multiple of 32 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q8_0(quantize_q8_0(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q8_0(tensor) => dequantize_q8_0(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ8_0Provider expects Q8_0 data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "q8_0"
    }
}

/// Compatibility provider that exports native Rust Q8_0 in GGML layout.
/// The external GGML quantizer is available in `quantinfer-adapter-ggml`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GgmlQ8_0Provider;

impl QuantizationProvider for GgmlQ8_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        NativeQ8_0Provider.scheme()
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        NativeQ8_0Provider.storage_descriptor(shape)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        NativeQ8_0Provider.validate(input)
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        NativeQ8_0Provider.quantize(input)
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        NativeQ8_0Provider.dequantize(input)
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q8_0"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ4_0Provider;

impl QuantizationProvider for NativeQ4_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_0
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q4_0,
            QuantizationScheme::Q4_0,
            shape,
            Q4_0_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        let values = input.as_f32()?;
        validate_finite(values)?;
        if !values.len().is_multiple_of(Q4_0_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q4_0 requires a multiple of 32 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q4_0(quantize_q4_0(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q4_0(tensor) => dequantize_q4_0(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ4_0Provider expects Q4_0 data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "q4_0"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ4KProvider;

impl QuantizationProvider for NativeQ4KProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_K
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q4_K,
            QuantizationScheme::Q4_K,
            shape,
            Q4_K_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        let values = input.as_f32()?;
        validate_finite(values)?;
        if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q4_K requires a multiple of 256 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q4_K(quantize_q4_k(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q4_K(tensor) => dequantize_q4_k(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ4KProvider expects Q4_K data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "q4_k"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ5KProvider;

impl QuantizationProvider for NativeQ5KProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q5_K
    }
    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q5_K,
            QuantizationScheme::Q5_K,
            shape,
            Q4_K_BLOCK_SIZE,
        )?)
    }
    fn validate(&self, input: &Tensor) -> Result<()> {
        let values = input.as_f32()?;
        validate_finite(values)?;
        if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q5_K requires a multiple of 256 elements".to_owned(),
            ));
        }
        Ok(())
    }
    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q5_K(quantize_q5_k(input)?))
    }
    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q5_K(tensor) => dequantize_q5_k(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ5KProvider expects Q5_K data".to_owned(),
            )),
        }
    }
    fn export_layout(&self) -> &'static str {
        "q5_k"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ6KProvider;

impl QuantizationProvider for NativeQ6KProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q6_K
    }
    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q6_K,
            QuantizationScheme::Q6_K,
            shape,
            Q4_K_BLOCK_SIZE,
        )?)
    }
    fn validate(&self, input: &Tensor) -> Result<()> {
        let values = input.as_f32()?;
        validate_finite(values)?;
        if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q6_K requires a multiple of 256 elements".to_owned(),
            ));
        }
        Ok(())
    }
    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q6_K(quantize_q6_k(input)?))
    }
    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q6_K(tensor) => dequantize_q6_k(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ6KProvider expects Q6_K data".to_owned(),
            )),
        }
    }
    fn export_layout(&self) -> &'static str {
        "q6_k"
    }
}
