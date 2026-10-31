//! Weight-only quantization primitives.

#![forbid(unsafe_code)]

use std::fmt;

use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor, Q8_0_BLOCK_SIZE};

pub type Result<T> = std::result::Result<T, QuantizationError>;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationError {
    InvalidInput(String),
    Core(quantinfer_core::CoreError),
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for QuantizationError {}

impl From<quantinfer_core::CoreError> for QuantizationError {
    fn from(error: quantinfer_core::CoreError) -> Self {
        Self::Core(error)
    }
}

pub fn quantize_q8_0(input: &Tensor) -> Result<QuantizedTensor> {
    let values = input.as_f32()?;
    if values.len() % Q8_0_BLOCK_SIZE != 0 {
        return Err(QuantizationError::InvalidInput(
            "Q8_0 requires a multiple of 32 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q8_0_BLOCK_SIZE);
    for chunk in values.chunks_exact(Q8_0_BLOCK_SIZE) {
        let maximum = chunk.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        let mut quantized = [0_i8; Q8_0_BLOCK_SIZE];
        for (index, value) in chunk.iter().copied().enumerate() {
            let rounded = (value / scale).round().clamp(-127.0, 127.0);
            quantized[index] = rounded as i8;
        }
        blocks.push(Q8_0Block {
            scale,
            values: quantized,
        });
    }
    Ok(QuantizedTensor::from_blocks(input.shape().clone(), blocks)?)
}

pub fn dequantize_q8_0(input: &QuantizedTensor) -> Result<Tensor> {
    let values = input
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(move |value| block.scale * f32::from(*value))
        })
        .collect();
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}

#[cfg(test)]
mod tests {
    use super::{dequantize_q8_0, quantize_q8_0};
    use quantinfer_core::Tensor;

    #[test]
    fn quantizes_and_dequantizes_a_block() {
        let input =
            Tensor::from_f32(vec![32], (0..32).map(|value| value as f32).collect()).unwrap();
        let quantized = quantize_q8_0(&input).unwrap();
        assert_eq!(quantized.blocks().len(), 1);
        let restored = dequantize_q8_0(&quantized).unwrap();
        let values = restored.as_f32().unwrap();
        assert!((values[31] - 31.0).abs() < 0.3);
    }

    #[test]
    fn rejects_non_block_aligned_input() {
        let input = Tensor::from_f32(vec![31], vec![1.0; 31]).unwrap();
        assert!(quantize_q8_0(&input).is_err());
    }
}
