//! Fixed block storage precision without introducing format I/O dependencies.
use half::f16;
use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
};

use super::QuantizedData;
use crate::{schemes::validate_finite, QuantizationError, Result};

fn stored(value: f32) -> Result<f32> {
    let rounded = f16::from_f32(value).to_f32();
    if !value.is_finite() || !rounded.is_finite() {
        return Err(QuantizationError::InvalidInput(
            "block scale cannot be represented as finite F16 storage".into(),
        ));
    }
    Ok(rounded)
}

impl QuantizedData {
    /// Copy the data with the precision used by the current fixed encodings.
    ///
    /// Integer codes are preserved. Quantized scale/min values round to F16,
    /// including underflow to zero, exactly as in GGUF/current QIF storage.
    /// Float data remains F32. Arbitrary Core block constructors and the source
    /// object retain their original F32 scales. This does not pack bytes or
    /// verify a published artifact, and does not reselect quantization codes.
    pub fn with_storage_precision(&self) -> Result<Self> {
        Ok(match self {
            Self::Float(tensor) => {
                validate_finite(tensor.as_f32()?)?;
                Self::Float(tensor.clone())
            }
            Self::Q8_0(tensor) => {
                let mut blocks = tensor.blocks().to_vec();
                for block in &mut blocks {
                    block.scale = stored(block.scale)?;
                }
                Self::Q8_0(QuantizedTensor::from_blocks(
                    tensor.shape().clone(),
                    blocks,
                )?)
            }
            Self::Q4_0(tensor) => {
                let mut blocks = tensor.blocks().to_vec();
                for block in &mut blocks {
                    block.scale = stored(block.scale)?;
                }
                Self::Q4_0(QuantizedTensor4::from_blocks(
                    tensor.shape().clone(),
                    blocks,
                )?)
            }
            Self::Q4_K(tensor) => {
                let mut blocks = tensor.blocks().to_vec();
                for block in &mut blocks {
                    block.scale = stored(block.scale)?;
                    block.min_scale = stored(block.min_scale)?;
                }
                Self::Q4_K(QuantizedTensorQ4K::from_blocks(
                    tensor.shape().clone(),
                    blocks,
                )?)
            }
            Self::Q5_K(tensor) => {
                let mut blocks = tensor.blocks().to_vec();
                for block in &mut blocks {
                    block.scale = stored(block.scale)?;
                    block.min_scale = stored(block.min_scale)?;
                }
                Self::Q5_K(QuantizedTensorQ5K::from_blocks(
                    tensor.shape().clone(),
                    blocks,
                )?)
            }
            Self::Q6_K(tensor) => {
                let mut blocks = tensor.blocks().to_vec();
                for block in &mut blocks {
                    block.scale = stored(block.scale)?;
                }
                Self::Q6_K(QuantizedTensorQ6K::from_blocks(
                    tensor.shape().clone(),
                    blocks,
                )?)
            }
        })
    }
}
