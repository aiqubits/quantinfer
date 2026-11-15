//! Q4_0 quantization provider.

use quantinfer_core::{
    DType, QuantizationScheme, Tensor, TensorStorageDescriptor, Q4_0_BLOCK_SIZE,
};

use super::{QuantizationProvider, QuantizedData};
use crate::schemes::{dequantize_q4_0, quantize_q4_0, validate_finite};
use crate::{QuantizationError, Result};

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
