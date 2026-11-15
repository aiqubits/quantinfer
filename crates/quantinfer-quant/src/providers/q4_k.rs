//! NativeQ4KProvider implementation.

use quantinfer_core::{
    DType, QuantizationScheme, Tensor, TensorStorageDescriptor, Q4_K_BLOCK_SIZE,
};

use super::{QuantizationProvider, QuantizedData};
use crate::schemes::{dequantize_q4_k, quantize_q4_k, validate_finite};
use crate::{QuantizationError, Result};

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
