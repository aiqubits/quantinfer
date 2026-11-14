//! CPU matrix and quantized block helpers.

#[cfg(feature = "simd")]
use quantinfer_core::{Q4_0Block, Q8_0Block};
use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
    Shape, Tensor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_kernel::{KernelError, Result};

use super::config::CpuScalar;

impl CpuScalar {
    pub(crate) fn matmul_rows_with<F>(
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
        self.matmul_rows_accumulate(lhs_values, rows, inner, columns, |output, index, left| {
            let rhs_offset = index * columns;
            for (column, result) in output.iter_mut().enumerate() {
                *result += left * value(rhs_offset + column)?;
            }
            Ok(())
        })
    }

    pub(crate) fn matmul_rows_accumulate<F>(
        &self,
        lhs_values: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
        accumulate: F,
    ) -> Result<Vec<f32>>
    where
        F: Fn(&mut [f32], usize, f32) -> Result<()> + Sync,
    {
        if columns == 0 {
            return Err(KernelError::InvalidInput(
                "matmul output must have at least one column".to_owned(),
            ));
        }
        let compute_rows = |lhs: &[f32], output: &mut [f32]| -> Result<()> {
            for (lhs_row, output_row) in lhs
                .chunks_exact(inner)
                .zip(output.chunks_exact_mut(columns))
            {
                for (index, &left) in lhs_row.iter().enumerate() {
                    accumulate(output_row, index, left)?;
                }
            }
            Ok(())
        };

        let worker_count = self.config.threads.max(1).min(rows);
        let output_elements = rows.checked_mul(columns).ok_or_else(|| {
            KernelError::InvalidInput("matmul output size overflows usize".to_owned())
        })?;
        let mut output = vec![0.0; output_elements];
        if worker_count == 1 {
            compute_rows(lhs_values, &mut output)?;
            return Ok(output);
        }

        let rows_per_worker = rows.div_ceil(worker_count);
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::with_capacity(worker_count);
            for (lhs, output) in lhs_values
                .chunks(rows_per_worker * inner)
                .zip(output.chunks_mut(rows_per_worker * columns))
            {
                let compute = &compute_rows;
                handles.push(scope.spawn(move || compute(lhs, output)));
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| KernelError::InvalidInput("CPU worker panicked".to_owned()))??;
            }
            Ok(())
        })?;
        Ok(output)
    }

    pub(crate) fn matmul_rows(
        &self,
        lhs_values: &[f32],
        rhs_values: &[f32],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "simd")]
        if let Some(kernel) = self.config.isa.scaled_add() {
            return self.matmul_rows_accumulate(
                lhs_values,
                rows,
                inner,
                columns,
                |output, index, left| {
                    let start = index * columns;
                    kernel.apply(output, &rhs_values[start..start + columns], left);
                    Ok(())
                },
            );
        }
        self.matmul_rows_with(lhs_values, rows, inner, columns, |index| {
            Ok(rhs_values[index])
        })
    }

    pub(crate) fn matmul_output_rows<B, T>(
        &self,
        lhs: &Tensor,
        rhs_shape: &Shape,
        blocks: &[B],
        unpack: impl Fn(&B) -> (f32, &[T; 32]),
        convert: impl Fn(T) -> f32,
    ) -> Result<Tensor>
    where
        T: Copy,
    {
        let (rows, inner) = self.expect_matrix(lhs, "lhs")?;
        let &[input, outputs] = rhs_shape.dims() else {
            return Err(KernelError::InvalidInput(
                "expected rank-2 quantized matrix".to_owned(),
            ));
        };
        if input != inner || input == 0 || outputs == 0 || !input.is_multiple_of(32) {
            return Err(KernelError::InvalidInput(
                "invalid quantized output rows".to_owned(),
            ));
        }
        let row_blocks = inner / 32;
        let elements = rows.checked_mul(outputs).ok_or_else(|| {
            KernelError::InvalidInput("matmul output size overflows usize".to_owned())
        })?;
        let mut result = Vec::with_capacity(elements);
        for left in self.f32_values(lhs, "lhs")?.chunks_exact(inner) {
            let left = left.as_chunks::<32>().0;
            for row in blocks.chunks_exact(row_blocks) {
                let mut sum = 0.0;
                for (left, block) in left.iter().zip(row) {
                    let (scale, codes) = unpack(block);
                    for (&activation, &code) in left.iter().zip(codes) {
                        sum += activation * (scale * convert(code));
                    }
                }
                result.push(sum);
            }
        }
        Ok(Tensor::from_f32(vec![rows, outputs], result)?)
    }

    pub(crate) fn quantized_value(&self, tensor: &QuantizedTensor, index: usize) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q8_0_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q8_0 index is out of bounds".to_owned()))?;
        Ok(block.scale * f32::from(block.values[index % Q8_0_BLOCK_SIZE]))
    }

    pub(crate) fn quantized_value_q4_0(
        &self,
        tensor: &QuantizedTensor4,
        index: usize,
    ) -> Result<f32> {
        let block = tensor
            .blocks()
            .get(index / Q4_0_BLOCK_SIZE)
            .ok_or_else(|| KernelError::InvalidInput("Q4_0 index is out of bounds".to_owned()))?;
        Ok(block.scale * (f32::from(block.values[index % Q4_0_BLOCK_SIZE]) - 8.0))
    }

    pub(crate) fn quantized_value_q4_k(
        &self,
        tensor: &QuantizedTensorQ4K,
        index: usize,
    ) -> Result<f32> {
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

    pub(crate) fn quantized_value_q5_k(
        &self,
        tensor: &QuantizedTensorQ5K,
        index: usize,
    ) -> Result<f32> {
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

    pub(crate) fn quantized_value_q6_k(
        &self,
        tensor: &QuantizedTensorQ6K,
        index: usize,
    ) -> Result<f32> {
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

    #[cfg(feature = "simd")]
    pub(crate) fn matmul_output_rows_q8_simd(
        &self,
        lhs: &Tensor,
        rhs_shape: &Shape,
        blocks: &[Q8_0Block],
        dot: quantinfer_backend_cpu_simd::QuantizedDot,
    ) -> Result<Tensor> {
        let (rows, inner) = self.expect_matrix(lhs, "lhs")?;
        let &[input, outputs] = rhs_shape.dims() else {
            return Err(KernelError::InvalidInput(
                "expected rank-2 quantized matrix".to_owned(),
            ));
        };
        let row_blocks = inner / Q8_0_BLOCK_SIZE;
        if input != inner
            || input == 0
            || outputs == 0
            || !input.is_multiple_of(Q8_0_BLOCK_SIZE)
            || blocks.len() != outputs * row_blocks
        {
            return Err(KernelError::InvalidInput(
                "invalid quantized output rows".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(rows * outputs);
        for left in self.f32_values(lhs, "lhs")?.chunks_exact(inner) {
            let left = left.as_chunks::<32>().0;
            for row in blocks.chunks_exact(row_blocks) {
                result.push(
                    left.iter()
                        .zip(row)
                        .map(|(left, block)| dot.q8(left, block.scale, &block.values))
                        .sum(),
                );
            }
        }
        Ok(Tensor::from_f32(vec![rows, outputs], result)?)
    }

    #[cfg(feature = "simd")]
    pub(crate) fn matmul_output_rows_q4_simd(
        &self,
        lhs: &Tensor,
        rhs_shape: &Shape,
        blocks: &[Q4_0Block],
        dot: quantinfer_backend_cpu_simd::QuantizedDot,
    ) -> Result<Tensor> {
        let (rows, inner) = self.expect_matrix(lhs, "lhs")?;
        let &[input, outputs] = rhs_shape.dims() else {
            return Err(KernelError::InvalidInput(
                "expected rank-2 quantized matrix".to_owned(),
            ));
        };
        let row_blocks = inner / Q4_0_BLOCK_SIZE;
        if input != inner
            || input == 0
            || outputs == 0
            || !input.is_multiple_of(Q4_0_BLOCK_SIZE)
            || blocks.len() != outputs * row_blocks
        {
            return Err(KernelError::InvalidInput(
                "invalid quantized output rows".to_owned(),
            ));
        }
        let mut result = Vec::with_capacity(rows * outputs);
        for left in self.f32_values(lhs, "lhs")?.chunks_exact(inner) {
            let left = left.as_chunks::<32>().0;
            for row in blocks.chunks_exact(row_blocks) {
                result.push(
                    left.iter()
                        .zip(row)
                        .map(|(left, block)| dot.q4(left, block.scale, &block.values))
                        .sum(),
                );
            }
        }
        Ok(Tensor::from_f32(vec![rows, outputs], result)?)
    }
}
