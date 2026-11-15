//! NativeQ6KProvider implementation.

use quantinfer_core::{
    DType, QuantizationScheme, Tensor, TensorStorageDescriptor, Q4_K_BLOCK_SIZE,
};

use super::{QuantizationProvider, QuantizedData};
use crate::schemes::{dequantize_q6_k, quantize_q6_k, validate_finite};
use crate::{QuantizationError, Result};

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
