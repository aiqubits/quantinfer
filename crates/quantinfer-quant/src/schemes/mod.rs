//! Native quantization and dequantization schemes.

mod q4;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8;

#[cfg(test)]
mod k_grid_tests;

pub use q4::{dequantize_q4_0, quantize_q4_0};
pub use q4_k::{dequantize_q4_k, quantize_q4_k};
pub use q5_k::{dequantize_q5_k, quantize_q5_k};
pub use q6_k::{dequantize_q6_k, quantize_q6_k};
pub use q8::{dequantize_q8_0, quantize_q8_0};

pub(crate) fn validate_finite(values: &[f32]) -> crate::Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(crate::QuantizationError::InvalidInput(
            "quantization requires finite input values".to_owned(),
        ));
    }
    Ok(())
}

fn stored_f16_scale(value: f32) -> crate::Result<f32> {
    let mut stored = half::f16::from_f32(value).to_f32();
    if !stored.is_finite() {
        return Err(crate::QuantizationError::InvalidInput(
            "K-quant scale cannot be represented as finite F16".to_owned(),
        ));
    }
    // A small global quotient must not collapse the entire superblock when
    // nonzero F16 scales can still represent its local grids.
    if stored == 0.0 && value > 0.0 {
        stored = half::f16::from_bits(1).to_f32();
    }
    Ok(stored)
}
