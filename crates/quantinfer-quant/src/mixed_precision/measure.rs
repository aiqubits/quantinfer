//! Measured candidates for native row-major [input, output] linear weights.
use quantinfer_core::{QuantizationScheme, Tensor};

use super::{PrecisionCandidate, PrecisionGroup};
use crate::{
    schemes::validate_finite, NativeQ4KProvider, NativeQ4_0Provider, NativeQ5KProvider,
    NativeQ6KProvider, NativeQ8_0Provider, QuantizationError, QuantizationProvider, Result,
};

#[derive(Debug, Clone, PartialEq)]
pub struct UnavailablePrecision {
    pub scheme: QuantizationScheme,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionMeasurements {
    pub group: PrecisionGroup,
    pub unavailable: Vec<UnavailablePrecision>,
}

fn invalid(message: impl Into<String>) -> QuantizationError {
    QuantizationError::InvalidInput(message.into())
}

/// Measure native fixed encodings using their actual F16 scale precision.
///
/// Input shape is `[input_channels, output_channels]`. Optional channel weights
/// are mean squared input activations. The objective is the diagonal proxy
/// `sum_i importance[i] * sum_j (weight[i,j] - restored[i,j])^2`, without
/// correlations between channels or model task-accuracy claims. Unaligned or
/// nonrepresentable encodings are explicitly reported as unavailable. F32 is
/// included only if requested. No tensor is serialized or parsed here.
pub fn measure_linear_precision_candidates(
    tensor_name: impl Into<String>,
    input: &Tensor,
    schemes: &[QuantizationScheme],
    channel_weights: Option<&[f32]>,
) -> Result<PrecisionMeasurements> {
    let name = tensor_name.into();
    let &[channels, outputs] = input.shape().dims() else {
        return Err(invalid(
            "precision measurement needs a rank-2 native linear tensor",
        ));
    };
    if name.is_empty() || channels == 0 || outputs == 0 || schemes.is_empty() || schemes.len() > 6 {
        return Err(invalid(
            "precision measurement needs a name, nonempty shape and 1..=6 schemes",
        ));
    }
    let values = input.as_f32()?;
    validate_finite(values)?;
    if let Some(weights) = channel_weights {
        if weights.len() != channels
            || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !weights.iter().any(|v| *v > 0.0)
        {
            return Err(invalid("precision channel weights must match inputs and be finite, nonnegative and nonzero"));
        }
    }
    let mut unique = Vec::new();
    for &scheme in schemes {
        if scheme == QuantizationScheme::Mixed || unique.contains(&scheme) {
            return Err(invalid(
                "precision measurement schemes must be unique tensor encodings",
            ));
        }
        unique.push(scheme);
    }
    let mut result = PrecisionMeasurements {
        group: PrecisionGroup {
            tensor_names: vec![name],
            candidates: Vec::new(),
        },
        unavailable: Vec::new(),
    };
    for &scheme in schemes {
        let (block, bytes) = match scheme {
            QuantizationScheme::None => (1, 4),
            QuantizationScheme::Q8_0 => (32, 34),
            QuantizationScheme::Q4_0 => (32, 18),
            QuantizationScheme::Q4_K => (256, 144),
            QuantizationScheme::Q5_K => (256, 176),
            QuantizationScheme::Q6_K => (256, 210),
            QuantizationScheme::Mixed => unreachable!(),
        };
        if !values.len().is_multiple_of(block) {
            result.unavailable.push(UnavailablePrecision {
                scheme,
                reason: format!("requires a multiple of {block} elements"),
            });
            continue;
        }
        let encoded_bytes = u64::try_from(values.len() / block)
            .ok()
            .and_then(|count| count.checked_mul(bytes))
            .ok_or_else(|| invalid("precision payload byte count overflows u64"))?;
        if scheme == QuantizationScheme::None {
            result.group.candidates.push(PrecisionCandidate {
                scheme,
                encoded_bytes,
                error: 0.0,
            });
            continue;
        }
        let provider: &dyn QuantizationProvider = match scheme {
            QuantizationScheme::Q8_0 => &NativeQ8_0Provider,
            QuantizationScheme::Q4_0 => &NativeQ4_0Provider,
            QuantizationScheme::Q4_K => &NativeQ4KProvider,
            QuantizationScheme::Q5_K => &NativeQ5KProvider,
            QuantizationScheme::Q6_K => &NativeQ6KProvider,
            _ => unreachable!(),
        };
        let measured = (|| -> Result<f64> {
            let data = provider.quantize(input)?.with_storage_precision()?;
            let restored = provider.dequantize(&data)?;
            let mut error = 0.0;
            for (index, (&original, &restored)) in values.iter().zip(restored.as_f32()?).enumerate()
            {
                let difference = f64::from(original) - f64::from(restored);
                let weight = channel_weights.map_or(1.0, |w| f64::from(w[index / outputs]));
                error += difference * difference * weight;
            }
            if !error.is_finite() {
                return Err(invalid("precision reconstruction objective is not finite"));
            }
            Ok(error)
        })();
        match measured {
            Ok(error) => result.group.candidates.push(PrecisionCandidate {
                scheme,
                encoded_bytes,
                error,
            }),
            Err(error) => result.unavailable.push(UnavailablePrecision {
                scheme,
                reason: error.to_string(),
            }),
        }
    }
    if result.group.candidates.is_empty() {
        return Err(invalid(format!(
            "no usable precision candidates: {:?}",
            result.unavailable
        )));
    }
    Ok(result)
}
