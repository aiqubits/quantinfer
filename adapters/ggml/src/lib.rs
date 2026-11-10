//! Optional Q8_0 quantizer backed by upstream GGML.

#![cfg(feature = "external")]

use std::ffi::{c_int, c_void};
use std::mem::size_of;
use std::ptr;

use half::f16;
use quantinfer_core::{
    Q8_0Block, QuantizationScheme, QuantizedTensor, Shape, Tensor, TensorStorageDescriptor,
    Q8_0_BLOCK_SIZE,
};
use quantinfer_quant::{
    NativeQ8_0Provider, QuantizationError, QuantizationProvider, QuantizedData, Result,
};

const GGML_TYPE_Q8_0: c_int = 8;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct GgmlBlock {
    scale: u16,
    values: [i8; Q8_0_BLOCK_SIZE],
}

unsafe extern "C" {
    fn ggml_blck_size(kind: c_int) -> i64;
    fn ggml_type_size(kind: c_int) -> usize;
    fn ggml_quantize_chunk(
        kind: c_int,
        source: *const f32,
        destination: *mut c_void,
        start: i64,
        rows: i64,
        values_per_row: i64,
        importance: *const f32,
    ) -> usize;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GgmlQ8_0Provider;

impl QuantizationProvider for GgmlQ8_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q8_0
    }

    fn storage_descriptor(&self, shape: Shape) -> Result<TensorStorageDescriptor> {
        NativeQ8_0Provider.storage_descriptor(shape)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        NativeQ8_0Provider.validate(input)?;
        let values = input.as_f32()?;
        if i32::try_from(values.len()).is_err() || values.iter().any(|value| !value.is_finite()) {
            return Err(QuantizationError::InvalidInput(
                "GGML Q8_0 requires finite weights and at most i32::MAX elements".to_owned(),
            ));
        }
        for block in values.as_chunks::<Q8_0_BLOCK_SIZE>().0 {
            let maximum = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
            let scale = maximum / 127.0;
            if scale > f16::MAX.to_f32() || (scale != 0.0 && !scale.recip().is_finite()) {
                return Err(QuantizationError::InvalidInput(
                    "GGML Q8_0 block scale is outside its representable range".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        let values = input.as_f32()?;
        let mut blocks = vec![GgmlBlock::default(); values.len() / Q8_0_BLOCK_SIZE];
        let expected_bytes = size_of::<GgmlBlock>() * blocks.len();
        if !blocks.is_empty() {
            let block_width = Q8_0_BLOCK_SIZE as i64;
            // SAFETY: these functions query the block layout for the valid Q8_0 enum value.
            let (width, bytes) = unsafe {
                (
                    ggml_blck_size(GGML_TYPE_Q8_0),
                    ggml_type_size(GGML_TYPE_Q8_0),
                )
            };
            if width != block_width || bytes != size_of::<GgmlBlock>() {
                return Err(QuantizationError::InvalidInput(
                    "incompatible GGML Q8_0 block layout".to_owned(),
                ));
            }
            // SAFETY: each row is one block, and validated weights occupy exactly rows times
            // block_width elements. The aligned destination has room for all rows. Q8_0
            // accepts a null importance matrix, and neither buffer is retained by GGML.
            let written = unsafe {
                ggml_quantize_chunk(
                    GGML_TYPE_Q8_0,
                    values.as_ptr(),
                    blocks.as_mut_ptr().cast(),
                    0,
                    blocks.len() as i64,
                    block_width,
                    ptr::null(),
                )
            };
            if written != expected_bytes {
                return Err(QuantizationError::InvalidInput(
                    "GGML Q8_0 returned an unexpected output size".to_owned(),
                ));
            }
        }
        let blocks = blocks
            .into_iter()
            .map(|block| Q8_0Block {
                scale: f16::from_bits(block.scale).to_f32(),
                values: block.values,
            })
            .collect();
        Ok(QuantizedData::Q8_0(QuantizedTensor::from_blocks(
            input.shape().clone(),
            blocks,
        )?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        NativeQ8_0Provider.dequantize(input)
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q8_0"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantizes_with_upstream_ggml_and_returns_native_q8_blocks() {
        let mut values = vec![0.0; 32];
        values.extend((0..32).map(|index| (index as f32 - 16.0) / 16.0));
        let input = Tensor::from_f32(vec![2, 32], values).unwrap();
        let quantized = GgmlQ8_0Provider.quantize(&input).unwrap();
        let QuantizedData::Q8_0(ref tensor) = quantized else {
            panic!("expected Q8_0 blocks");
        };
        assert_eq!(tensor.blocks()[0].scale, 0.0);
        assert_eq!(tensor.blocks()[1].values[0], -127);
        assert!(
            GgmlQ8_0Provider
                .error_report(&input, &quantized)
                .unwrap()
                .max_absolute_error
                < 0.005
        );
    }

    #[test]
    fn rejects_unaligned_or_non_finite_input_before_ffi() {
        for values in [vec![0.0; 31], vec![f32::NAN; 32], vec![f32::MAX; 32]] {
            let input = Tensor::from_f32(vec![values.len()], values).unwrap();
            assert!(GgmlQ8_0Provider.quantize(&input).is_err());
        }
    }
}
