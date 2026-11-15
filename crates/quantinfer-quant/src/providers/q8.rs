//! Q8_0 quantization providers.

use quantinfer_core::{
    DType, QuantizationScheme, Tensor, TensorStorageDescriptor, Q8_0_BLOCK_SIZE,
};

use super::{QuantizationProvider, QuantizedData};
use crate::schemes::{dequantize_q8_0, quantize_q8_0, validate_finite};
use crate::{QuantizationError, Result};

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
