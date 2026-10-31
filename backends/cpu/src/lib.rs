//! Reference CPU kernels implemented without a vendor runtime dependency.

#![forbid(unsafe_code)]

use quantinfer_core::{QuantizedTensor, Shape, Tensor, Q8_0_BLOCK_SIZE};
use quantinfer_kernel::{KernelBackend, KernelError, Result};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuScalar;

impl CpuScalar {
    pub const fn new() -> Self {
        Self
    }

    fn f32_values<'a>(&self, tensor: &'a Tensor, name: &str) -> Result<&'a [f32]> {
        tensor
            .as_f32()
            .map_err(|error| KernelError::InvalidInput(format!("{name} must be F32: {error}")))
    }

    fn expect_matrix(&self, tensor: &Tensor, name: &str) -> Result<(usize, usize)> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(KernelError::InvalidInput(format!(
                "{name} must be rank 2, got rank {}",
                dims.len()
            )));
        }
        Ok((dims[0], dims[1]))
    }

    fn expect_last_dim(&self, tensor: &Tensor, name: &str) -> Result<usize> {
        tensor.shape().last_dim().ok_or_else(|| {
            KernelError::InvalidInput(format!("{name} must have at least one dimension"))
        })
    }

    fn quantized_value(&self, tensor: &QuantizedTensor, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q8_0_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q8_0 index is out of bounds".to_owned()))?;
        Ok(block.scale * f32::from(block.values[index % Q8_0_BLOCK_SIZE]))
    }
}

impl KernelBackend for CpuScalar {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let (rhs_k, n) = self.expect_matrix(rhs, "rhs")?;
        if k != rhs_k {
            return Err(KernelError::InvalidInput(format!(
                "matmul dimensions do not align: lhs is {m}x{k}, rhs is {rhs_k}x{n}"
            )));
        }
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let rhs_values = self.f32_values(rhs, "rhs")?;
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += lhs_values[row * k + inner] * rhs_values[inner * n + column];
                }
                output[row * n + column] = sum;
            }
        }
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn matmul_q8_0(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let rhs_shape = rhs.shape().dims();
        if rhs_shape.len() != 2 || rhs_shape[0] != k {
            return Err(KernelError::InvalidInput(
                "Q8_0 rhs must be a rank-2 matrix with a matching inner dimension".to_owned(),
            ));
        }
        let n = rhs_shape[1];
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    let rhs_index = inner * n + column;
                    sum += lhs_values[row * k + inner] * self.quantized_value(rhs, rhs_index)?;
                }
                output[row * n + column] = sum;
            }
        }
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        let last_dim = self.expect_last_dim(input, "input")?;
        let weight_values = self.f32_values(weight, "weight")?;
        if weight.shape().dims() != [last_dim] || weight_values.len() != last_dim {
            return Err(KernelError::InvalidInput(
                "RMSNorm weight must match the input last dimension".to_owned(),
            ));
        }
        let input_values = self.f32_values(input, "input")?;
        let mut output = vec![0.0; input_values.len()];
        for row in 0..input_values.len() / last_dim {
            let start = row * last_dim;
            let values = &input_values[start..start + last_dim];
            let mean_square =
                values.iter().map(|value| value * value).sum::<f32>() / last_dim as f32;
            let inverse = (mean_square + epsilon).sqrt().recip();
            for column in 0..last_dim {
                output[start + column] = values[column] * inverse * weight_values[column];
            }
        }
        Ok(Tensor::from_f32(input.shape().clone(), output)?)
    }

    fn rms_norm_heads(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 3 {
            return Err(KernelError::InvalidInput(
                "head RMSNorm input must have shape [tokens, heads, head_dim]".to_owned(),
            ));
        }
        self.rms_norm(input, weight, epsilon)
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        let last_dim = self.expect_last_dim(input, "input")?;
        let input_values = self.f32_values(input, "input")?;
        let mut output = vec![0.0; input_values.len()];
        for row in 0..input_values.len() / last_dim {
            let start = row * last_dim;
            let values = &input_values[start..start + last_dim];
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = values
                .iter()
                .map(|value| (*value - maximum).exp())
                .sum::<f32>();
            for column in 0..last_dim {
                output[start + column] = (values[column] - maximum).exp() / sum;
            }
        }
        Ok(Tensor::from_f32(input.shape().clone(), output)?)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        if gate.shape() != up.shape() {
            return Err(KernelError::InvalidInput(
                "SwiGLU inputs must have identical shapes".to_owned(),
            ));
        }
        let gate_values = self.f32_values(gate, "gate")?;
        let up_values = self.f32_values(up, "up")?;
        let output = gate_values
            .iter()
            .zip(up_values)
            .map(|(gate_value, up_value)| gate_value / (1.0 + (-gate_value).exp()) * up_value)
            .collect();
        Ok(Tensor::from_f32(gate.shape().clone(), output)?)
    }

    fn apply_rope(&self, input: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 3 || dims[2] % 2 != 0 {
            return Err(KernelError::InvalidInput(
                "RoPE input must have shape [tokens, heads, even_head_dim]".to_owned(),
            ));
        }
        let head_dim = dims[2];
        let pair_count = head_dim / 2;
        if cos.len() != pair_count && cos.len() != dims[0] * pair_count {
            return Err(KernelError::InvalidInput(
                "RoPE tables must have one row or one row per token".to_owned(),
            ));
        }
        if sin.len() != cos.len() {
            return Err(KernelError::InvalidInput(
                "RoPE cosine and sine tables must have identical lengths".to_owned(),
            ));
        }
        let input_values = self.f32_values(input, "input")?;
        let mut output = input_values.to_vec();
        for token in 0..dims[0] {
            for head in 0..dims[1] {
                let start = (token * dims[1] + head) * head_dim;
                let table_offset = if cos.len() == pair_count {
                    0
                } else {
                    token * pair_count
                };
                for pair in 0..pair_count {
                    let first = input_values[start + pair * 2];
                    let second = input_values[start + pair * 2 + 1];
                    output[start + pair * 2] =
                        first * cos[table_offset + pair] - second * sin[table_offset + pair];
                    output[start + pair * 2 + 1] =
                        first * sin[table_offset + pair] + second * cos[table_offset + pair];
                }
            }
        }
        Ok(Tensor::from_f32(input.shape().clone(), output)?)
    }

    fn embedding(&self, table: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
        let (vocabulary, hidden) = self.expect_matrix(table, "embedding table")?;
        let table_values = self.f32_values(table, "embedding table")?;
        let mut output = Vec::with_capacity(token_ids.len() * hidden);
        for token_id in token_ids {
            if *token_id >= vocabulary {
                return Err(KernelError::InvalidInput(format!(
                    "token id {token_id} is outside vocabulary size {vocabulary}"
                )));
            }
            output.extend_from_slice(&table_values[token_id * hidden..(token_id + 1) * hidden]);
        }
        Ok(Tensor::from_f32(
            Shape::from(vec![token_ids.len().max(1), hidden]),
            output,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuScalar, KernelBackend};
    use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor};

    #[test]
    fn computes_matrix_product() {
        let backend = CpuScalar::new();
        let lhs = Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Tensor::from_f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = backend.matmul(&lhs, &rhs).unwrap();
        assert_eq!(result.as_f32().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn computes_q8_matrix_product() {
        let backend = CpuScalar::new();
        let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
        let mut values = [0_i8; 32];
        values.fill(1);
        let rhs = QuantizedTensor::from_blocks(vec![32, 1], vec![Q8_0Block { scale: 2.0, values }])
            .unwrap();
        let result = backend.matmul_q8_0(&lhs, &rhs).unwrap();
        assert_eq!(result.as_f32().unwrap(), &[64.0]);
    }
}
