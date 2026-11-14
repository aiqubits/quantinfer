//! CPU KernelBackend operator implementations.

use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
    Shape, Tensor,
};
use quantinfer_kernel::{KernelBackend, KernelError, Result};

use super::config::CpuScalar;

impl KernelBackend for CpuScalar {
    fn reuse_workspace(&self) -> bool {
        self.config.workspace_reuse
    }

    fn matmul_q8_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        #[cfg(feature = "simd")]
        if let Some(dot) = self.config.isa.quantized_dot() {
            return self.matmul_output_rows_q8_simd(lhs, rhs.shape(), rhs.blocks(), dot);
        }
        self.matmul_output_rows(
            lhs,
            rhs.shape(),
            rhs.blocks(),
            |block| (block.scale, &block.values),
            f32::from,
        )
    }

    fn matmul_q4_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        #[cfg(feature = "simd")]
        if let Some(dot) = self.config.isa.quantized_dot() {
            return self.matmul_output_rows_q4_simd(lhs, rhs.shape(), rhs.blocks(), dot);
        }
        self.matmul_output_rows(
            lhs,
            rhs.shape(),
            rhs.blocks(),
            |block| (block.scale, &block.values),
            |code| f32::from(code) - 8.0,
        )
    }

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
        let output = self.matmul_rows(lhs_values, rhs_values, m, k, n)?;
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
        let output = self.matmul_rows_with(lhs_values, m, k, n, |index| {
            self.quantized_value(rhs, index)
        })?;
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let rhs_shape = rhs.shape().dims();
        if rhs_shape.len() != 2 || rhs_shape[0] != k {
            return Err(KernelError::InvalidInput(
                "Q4_0 rhs must be a rank-2 matrix with a matching inner dimension".to_owned(),
            ));
        }
        let n = rhs_shape[1];
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let output = self.matmul_rows_with(lhs_values, m, k, n, |index| {
            self.quantized_value_q4_0(rhs, index)
        })?;
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn matmul_q4_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let rhs_shape = rhs.shape().dims();
        if rhs_shape.len() != 2 || rhs_shape[0] != k {
            return Err(KernelError::InvalidInput(
                "Q4_K rhs must be a rank-2 matrix with a matching inner dimension".to_owned(),
            ));
        }
        let n = rhs_shape[1];
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let output = self.matmul_rows_with(lhs_values, m, k, n, |index| {
            self.quantized_value_q4_k(rhs, index)
        })?;
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn matmul_q5_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let rhs_shape = rhs.shape().dims();
        if rhs_shape.len() != 2 || rhs_shape[0] != k {
            return Err(KernelError::InvalidInput(
                "Q5_K rhs must be a rank-2 matrix with a matching inner dimension".to_owned(),
            ));
        }
        let n = rhs_shape[1];
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let output = self.matmul_rows_with(lhs_values, m, k, n, |index| {
            self.quantized_value_q5_k(rhs, index)
        })?;
        Ok(Tensor::from_f32(vec![m, n], output)?)
    }

    fn matmul_q6_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> Result<Tensor> {
        let (m, k) = self.expect_matrix(lhs, "lhs")?;
        let rhs_shape = rhs.shape().dims();
        if rhs_shape.len() != 2 || rhs_shape[0] != k {
            return Err(KernelError::InvalidInput(
                "Q6_K rhs must be a rank-2 matrix with a matching inner dimension".to_owned(),
            ));
        }
        let n = rhs_shape[1];
        let lhs_values = self.f32_values(lhs, "lhs")?;
        let output = self.matmul_rows_with(lhs_values, m, k, n, |index| {
            self.quantized_value_q6_k(rhs, index)
        })?;
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
        if dims.len() != 3 || !dims[2].is_multiple_of(2) {
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
                    let first = input_values[start + pair];
                    let second = input_values[start + pair_count + pair];
                    output[start + pair] =
                        first * cos[table_offset + pair] - second * sin[table_offset + pair];
                    output[start + pair_count + pair] =
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
