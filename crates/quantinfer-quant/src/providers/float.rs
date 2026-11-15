//! Floating-point quantization provider.

use quantinfer_core::{DType, QuantizationScheme, Tensor, TensorStorageDescriptor};

use super::{QuantizationProvider, QuantizedData};
use crate::{QuantizationError, Result};

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
