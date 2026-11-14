use half::f16;
use quantinfer_core::{
    Q4_0Block, QuantizationAxis, QuantizationScheme, QuantizedTensor4, Shape, Tensor,
    TensorStorageDescriptor, OUTPUT_ROWS_LAYOUT, Q4_0_BLOCK_SIZE,
};

use crate::{NativeQ4_0Provider, QuantizationError, QuantizationProvider, QuantizedData, Result};

/// Calibrate an output-row matrix using mean-square input activations.
/// The logical tensor shape is [input, output]. Data is stored by output row.
pub struct ImatrixQ4_0Provider<'a> {
    importance: &'a [f32],
}

impl<'a> ImatrixQ4_0Provider<'a> {
    pub fn new(importance: &'a [f32]) -> Result<Self> {
        if !importance.len().is_multiple_of(Q4_0_BLOCK_SIZE)
            || !importance.iter().any(|&weight| weight > 0.0)
            || importance
                .iter()
                .any(|&weight| !weight.is_finite() || weight < 0.0)
        {
            return Err(invalid("imatrix weights must be finite, nonnegative, positive in at least one channel, and block aligned"));
        }
        Ok(Self { importance })
    }

    fn check_shape(&self, shape: &Shape) -> Result<()> {
        if !matches!(shape.dims(), [width, _] if *width == self.importance.len()) {
            return Err(invalid(
                "imatrix channel count must match the matrix input width",
            ));
        }
        Ok(())
    }
}

impl QuantizationProvider for ImatrixQ4_0Provider<'_> {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_0
    }

    fn storage_descriptor(&self, shape: Shape) -> Result<TensorStorageDescriptor> {
        self.check_shape(&shape)?;
        let mut descriptor = NativeQ4_0Provider.storage_descriptor(shape)?;
        descriptor.physical_layout = OUTPUT_ROWS_LAYOUT.to_owned();
        descriptor.quantization_axis = QuantizationAxis::Axis(0);
        Ok(descriptor)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        self.check_shape(input.shape())?;
        if input.as_f32()?.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "imatrix quantization requires finite source weights",
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        let width = self.importance.len();
        let values = input.as_f32()?;
        let mut blocks = Vec::with_capacity(values.len() / Q4_0_BLOCK_SIZE);
        for row in values.chunks_exact(width) {
            for (weights, importance) in row
                .as_chunks::<Q4_0_BLOCK_SIZE>()
                .0
                .iter()
                .zip(self.importance.as_chunks::<Q4_0_BLOCK_SIZE>().0)
            {
                blocks.push(fit_block(weights, importance));
            }
        }
        Ok(QuantizedData::Q4_0(QuantizedTensor4::from_blocks(
            input.shape().clone(),
            blocks,
        )?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        NativeQ4_0Provider.dequantize(input)
    }

    fn export_layout(&self) -> &'static str {
        NativeQ4_0Provider.export_layout()
    }
}

fn fit_block(weights: &[f32; Q4_0_BLOCK_SIZE], importance: &[f32; Q4_0_BLOCK_SIZE]) -> Q4_0Block {
    let mut magnitudes = weights.map(f32::abs);
    magnitudes.sort_unstable_by(f32::total_cmp);
    let max = magnitudes[Q4_0_BLOCK_SIZE - 1];
    let mut best = encode_block(weights, max / 7.0);
    let mut best_error = weighted_error(&best, weights, importance);
    for index in [30, 28, 24, 16] {
        let candidate = encode_block(weights, magnitudes[index] / 7.0);
        let error = weighted_error(&candidate, weights, importance);
        if error < best_error {
            best = candidate;
            best_error = error;
        }
    }
    best
}

fn encode_block(weights: &[f32; Q4_0_BLOCK_SIZE], scale: f32) -> Q4_0Block {
    // Fit the actual binary16 scale that GGUF will store.
    let scale = f16::from_f32(scale.min(f16::MAX.to_f32())).to_f32();
    let values = if scale == 0.0 {
        [8; Q4_0_BLOCK_SIZE]
    } else {
        weights.map(|weight| ((weight / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8)
    };
    Q4_0Block { scale, values }
}

fn weighted_error(block: &Q4_0Block, weights: &[f32], importance: &[f32]) -> f64 {
    block
        .values
        .iter()
        .zip(weights)
        .zip(importance)
        .map(|((&quantized, &weight), &importance)| {
            let restored = f64::from(block.scale) * (f64::from(quantized) - 8.0);
            let error = f64::from(weight) - restored;
            f64::from(importance) * error * error
        })
        .sum()
}

fn invalid(message: &str) -> QuantizationError {
    QuantizationError::InvalidInput(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prioritizes_active_channels_in_each_output_row() {
        let mut row = std::array::from_fn::<_, 32, _>(|channel| (channel % 15) as f32 - 7.0);
        row[31] = 120.0;
        let mut importance = [1.0; 32];
        importance[31] = 0.0;
        let source = Tensor::from_f32(vec![32, 2], [row, row].concat()).unwrap();
        let provider = ImatrixQ4_0Provider::new(&importance).unwrap();
        let calibrated = provider.quantize(&source).unwrap();
        let decoded = provider.dequantize(&calibrated).unwrap();
        let values = decoded.as_f32().unwrap();
        assert_eq!(values[..31], row[..31]);
        assert_eq!(values[32..63], row[..31]);
        assert_ne!(calibrated, NativeQ4_0Provider.quantize(&source).unwrap());
        assert!(provider
            .quantize(&source.reshape(vec![16, 4]).unwrap())
            .is_err());
        assert!(ImatrixQ4_0Provider::new(&[0.0; 32]).is_err());
    }
}
