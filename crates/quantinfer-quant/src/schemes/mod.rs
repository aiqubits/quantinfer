//! Native quantization and dequantization schemes.

mod q4;
mod q4_k;
mod q5_k;
mod q6_k;
mod q8;

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
