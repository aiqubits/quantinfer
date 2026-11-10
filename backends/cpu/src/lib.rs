//! Reference CPU kernels implemented without a vendor runtime dependency.

#![forbid(unsafe_code)]

use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
    Shape, Tensor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_kernel::{KernelBackend, KernelError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIsa {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl CpuIsa {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            return CpuIsa::Neon;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return CpuIsa::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return CpuIsa::Avx2;
            }
        }
        CpuIsa::Scalar
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuRuntimeConfig {
    pub isa: CpuIsa,
    pub threads: usize,
    pub workspace_reuse: bool,
}

impl Default for CpuRuntimeConfig {
    fn default() -> Self {
        Self {
            isa: CpuIsa::detect(),
            threads: 1,
            workspace_reuse: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuScalar {
    config: CpuRuntimeConfig,
}

impl Default for CpuScalar {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuScalar {
    pub const fn new() -> Self {
        Self {
            config: CpuRuntimeConfig {
                isa: CpuIsa::Scalar,
                threads: 1,
                workspace_reuse: true,
            },
        }
    }

    pub const fn with_config(config: CpuRuntimeConfig) -> Self {
        Self { config }
    }

    pub const fn config(&self) -> CpuRuntimeConfig {
        self.config
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

    fn matmul_rows_with<F>(
        &self,
        lhs_values: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
        value: F,
    ) -> Result<Vec<f32>>
    where
        F: Fn(usize) -> Result<f32> + Sync + Send,
    {
        let worker_count = self.config.threads.max(1).min(rows.max(1));
        let mut output = vec![0.0; rows * columns];
        if worker_count == 1 || rows < 2 {
            for row in 0..rows {
                for column in 0..columns {
                    let mut sum = 0.0;
                    for index in 0..inner {
                        sum += lhs_values[row * inner + index] * value(index * columns + column)?;
                    }
                    output[row * columns + column] = sum;
                }
            }
            return Ok(output);
        }

        let chunk_size = rows.div_ceil(worker_count);
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for start in (0..rows).step_by(chunk_size) {
                let end = (start + chunk_size).min(rows);
                let value_ref = &value;
                handles.push(scope.spawn(move || {
                    let mut values = vec![0.0; (end - start) * columns];
                    for row in start..end {
                        for column in 0..columns {
                            let mut sum = 0.0;
                            for index in 0..inner {
                                sum += lhs_values[row * inner + index]
                                    * value_ref(index * columns + column)?;
                            }
                            values[(row - start) * columns + column] = sum;
                        }
                    }
                    Ok::<_, KernelError>((start, values))
                }));
            }
            for handle in handles {
                let (start, values) = handle
                    .join()
                    .map_err(|_| KernelError::InvalidInput("CPU worker panicked".to_owned()))??;
                output[start * columns..(start * columns) + values.len()].copy_from_slice(&values);
            }
            Ok(())
        })?;
        Ok(output)
    }

    fn matmul_rows(
        &self,
        lhs_values: &[f32],
        rhs_values: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        self.matmul_rows_with(lhs_values, rows, inner, columns, |index| {
            Ok(rhs_values[index])
        })
    }

    fn quantized_value(&self, tensor: &QuantizedTensor, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q8_0_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q8_0 index is out of bounds".to_owned()))?;
        Ok(block.scale * f32::from(block.values[index % Q8_0_BLOCK_SIZE]))
    }

    fn quantized_value_q4_0(&self, tensor: &QuantizedTensor4, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q4_0_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q4_0 index is out of bounds".to_owned()))?;
        Ok(block.scale * (f32::from(block.values[index % Q4_0_BLOCK_SIZE]) - 8.0))
    }

    fn quantized_value_q4_k(&self, tensor: &QuantizedTensorQ4K, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q4_K_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q4_K index is out of bounds".to_owned()))?;
        let local = index % Q4_K_BLOCK_SIZE;
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for subblock in 0..4 {
            scales[subblock] = block.scales[subblock] & 0x3f;
            mins[subblock] = block.scales[4 + subblock] & 0x3f;
            scales[4 + subblock] =
                (block.scales[8 + subblock] & 0x0f) | ((block.scales[subblock] >> 6) << 4);
            mins[4 + subblock] =
                (block.scales[8 + subblock] >> 4) | ((block.scales[4 + subblock] >> 6) << 4);
        }
        let group = local / 64;
        let within_group = local % 64;
        let packed = block.values[group * 32 + within_group % 32];
        let quantized = if within_group < 32 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        let subblock = local / 32;
        Ok(
            block.scale * f32::from(scales[subblock]) * f32::from(quantized)
                - block.min_scale * f32::from(mins[subblock]),
        )
    }

    fn quantized_value_q5_k(&self, tensor: &QuantizedTensorQ5K, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q4_K_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q5_K index is out of bounds".to_owned()))?;
        let local = index % Q4_K_BLOCK_SIZE;
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for subblock in 0..4 {
            scales[subblock] = block.scales[subblock] & 0x3f;
            mins[subblock] = block.scales[4 + subblock] & 0x3f;
            scales[4 + subblock] =
                (block.scales[8 + subblock] & 0xf) | ((block.scales[subblock] >> 6) << 4);
            mins[4 + subblock] =
                (block.scales[8 + subblock] >> 4) | ((block.scales[4 + subblock] >> 6) << 4);
        }
        let group = local / 64;
        let within = local % 64;
        let packed = block.values[group * 32 + within % 32];
        let mut q = if within < 32 {
            packed & 0xf
        } else {
            packed >> 4
        };
        if block.high_bits[within % 32]
            & (1 << (if within < 32 {
                2 * group
            } else {
                2 * group + 1
            }))
            != 0
        {
            q += 16;
        }
        let subblock = local / 32;
        Ok(block.scale * f32::from(scales[subblock]) * f32::from(q)
            - block.min_scale * f32::from(mins[subblock]))
    }

    fn quantized_value_q6_k(&self, tensor: &QuantizedTensorQ6K, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q4_K_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q6_K index is out of bounds".to_owned()))?;
        let local = index % Q4_K_BLOCK_SIZE;
        let ip = local / 128;
        let within = local % 128;
        let chunk = within / 32;
        let il = within % 32;
        let packed = block.low_bits[ip * 64 + if chunk.is_multiple_of(2) { il } else { 32 + il }];
        let low = if chunk < 2 { packed & 0xf } else { packed >> 4 };
        let high = block.high_bits[ip * 32 + il];
        let q = low | (((high >> (chunk * 2)) & 3) << 4);
        let subblock = local / 16;
        Ok(block.scale * f32::from(block.scales[subblock]) * (i32::from(q) - 32) as f32)
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

#[cfg(test)]
mod tests {
    use super::{CpuIsa, CpuRuntimeConfig, CpuScalar, KernelBackend};
    use quantinfer_core::{
        Q4KBlock, Q4_0Block, Q5KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
        QuantizedTensorQ4K, QuantizedTensorQ5K, Tensor,
    };

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

    #[test]
    fn computes_q4_matrix_product() {
        let backend = CpuScalar::new();
        let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
        let rhs = QuantizedTensor4::from_blocks(
            vec![32, 1],
            vec![Q4_0Block {
                scale: 1.0,
                values: [8_u8; 32],
            }],
        )
        .unwrap();
        let result = backend.matmul_q4_0(&lhs, &rhs).unwrap();
        assert_eq!(result.as_f32().unwrap(), &[0.0]);
    }

    #[test]
    fn computes_q4_k_matrix_product() {
        let backend = CpuScalar::new();
        let lhs = Tensor::from_f32(vec![1, 256], vec![1.0; 256]).unwrap();
        let rhs = QuantizedTensorQ4K::from_blocks(
            vec![256, 1],
            vec![Q4KBlock {
                scale: 0.1,
                min_scale: 0.0,
                scales: [63; 12],
                values: [0x99; 128],
            }],
        )
        .unwrap();
        let result = backend.matmul_q4_k(&lhs, &rhs).unwrap();
        assert!(result.as_f32().unwrap()[0] > 200.0);
    }

    #[test]
    fn computes_q5_k_matrix_product() {
        let backend = CpuScalar::new();
        let lhs = Tensor::from_f32(vec![1, 256], vec![1.0; 256]).unwrap();
        let rhs = QuantizedTensorQ5K::from_blocks(
            vec![256, 1],
            vec![Q5KBlock {
                scale: 0.1,
                min_scale: 0.0,
                scales: [63; 12],
                high_bits: [0; 32],
                values: [0x99; 128],
            }],
        )
        .unwrap();
        let result = backend.matmul_q5_k(&lhs, &rhs).unwrap();
        assert!(result.as_f32().unwrap()[0] > 200.0);
    }

    #[test]
    fn applies_half_split_rope_rotation() {
        let backend = CpuScalar::new();
        let input = Tensor::from_f32(vec![1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = backend
            .apply_rope(&input, &[0.0, 1.0], &[1.0, 0.0])
            .unwrap();
        assert_eq!(output.as_f32().unwrap(), &[-3.0, 2.0, 1.0, 4.0]);
    }

    #[test]
    fn exposes_safe_cpu_isa_and_thread_configuration() {
        let backend = CpuScalar::with_config(CpuRuntimeConfig {
            isa: CpuIsa::Scalar,
            threads: 4,
            workspace_reuse: true,
        });
        assert_eq!(backend.config().threads, 4);
        assert!(backend.config().workspace_reuse);
    }

    #[test]
    fn threaded_f32_matmul_matches_scalar_reference() {
        let lhs = Tensor::from_f32(
            vec![8, 4],
            (0..32).map(|index| index as f32 / 10.0).collect(),
        )
        .unwrap();
        let rhs = Tensor::from_f32(
            vec![4, 3],
            (0..12).map(|index| index as f32 / 7.0).collect(),
        )
        .unwrap();
        let scalar = CpuScalar::new().matmul(&lhs, &rhs).unwrap();
        let threaded = CpuScalar::with_config(CpuRuntimeConfig {
            isa: CpuIsa::Scalar,
            threads: 4,
            workspace_reuse: true,
        })
        .matmul(&lhs, &rhs)
        .unwrap();
        assert_eq!(threaded.as_f32().unwrap(), scalar.as_f32().unwrap());
    }

    #[test]
    fn threaded_q8_matmul_matches_scalar_reference() {
        let lhs = Tensor::from_f32(vec![8, 32], vec![1.0; 8 * 32]).unwrap();
        let mut values = [0_i8; 32];
        values.fill(1);
        let rhs = QuantizedTensor::from_blocks(
            vec![32, 1],
            vec![quantinfer_core::Q8_0Block { scale: 0.5, values }],
        )
        .unwrap();
        let scalar = CpuScalar::new().matmul_q8_0(&lhs, &rhs).unwrap();
        let threaded = CpuScalar::with_config(CpuRuntimeConfig {
            isa: CpuIsa::Scalar,
            threads: 4,
            workspace_reuse: true,
        })
        .matmul_q8_0(&lhs, &rhs)
        .unwrap();
        assert_eq!(threaded.as_f32().unwrap(), scalar.as_f32().unwrap());
    }
}
