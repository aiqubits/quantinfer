//! Safe F64 Cholesky operations for the full, damped calibration Hessian.

use super::{invalid, zeroes};
use quantinfer_quant::Result;

fn cholesky(matrix: &mut [f64], n: usize) -> Result<()> {
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j];
            for k in 0..j {
                value -= matrix[i * n + k] * matrix[j * n + k];
            }
            if !value.is_finite() || (i == j && value <= 0.0) {
                return Err(invalid(
                    "damped Hessian is not numerically positive definite; increase damping",
                ));
            }
            matrix[i * n + j] = if i == j {
                value.sqrt()
            } else {
                value / matrix[j * n + j]
            };
            if !matrix[i * n + j].is_finite() {
                return Err(invalid("Hessian factor exceeds finite F64 range"));
            }
        }
        matrix[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

/// Return upper U with U^T U = H^-1, as used by GPTQ error feedback.
/// This is not inverse(cholesky(H)): the triangular order matters.
pub(super) fn inverse_factor(mut matrix: Vec<f64>, n: usize) -> Result<Vec<f64>> {
    cholesky(&mut matrix, n)?;
    let mut inverse_lower = zeroes(matrix.len())?;
    for i in 0..n {
        inverse_lower[i * n + i] = 1.0 / matrix[i * n + i];
        for j in 0..i {
            let mut sum = 0.0;
            for k in j..i {
                sum += matrix[i * n + k] * inverse_lower[k * n + j];
            }
            inverse_lower[i * n + j] = -sum / matrix[i * n + i];
        }
    }
    for i in 0..n {
        for j in 0..=i {
            matrix[i * n + j] = (i..n)
                .map(|k| inverse_lower[k * n + i] * inverse_lower[k * n + j])
                .sum();
        }
    }
    drop(inverse_lower);
    cholesky(&mut matrix, n)?;
    for i in 0..n {
        for j in 0..i {
            matrix.swap(i * n + j, j * n + i);
        }
    }
    Ok(matrix)
}
