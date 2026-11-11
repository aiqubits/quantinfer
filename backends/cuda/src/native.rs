use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K,
    Tensor,
};
use quantinfer_gpu_common::{projection_capabilities, ProjectionWeights};
use quantinfer_kernel::{KernelBackend, KernelError, Result};
use quantinfer_runtime::{KernelCapability, RuntimeProvider};

const TILE: u32 = 16;
const KERNEL_SOURCE: &str = include_str!("../kernels/matmul.cu");

/// Tiled F32, Q4_0, and Q8_0 projections on an NVIDIA CUDA device.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    name: String,
    capabilities: Vec<KernelCapability>,
    cpu: CpuScalar,
}

impl CudaBackend {
    /// Open a CUDA device by its visible ordinal and compile the projection kernel.
    pub fn open(ordinal: usize) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unavailable(
                "packed projections require a little-endian host",
            ));
        }
        // The probe only loads and releases a driver library from the host search path.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(unavailable("the NVIDIA CUDA driver library is unavailable"));
        }
        let context = CudaContext::new(ordinal).map_err(unavailable)?;
        if context.compute_capability().map_err(unavailable)? < (7, 5) {
            return Err(unavailable("compute capability 7.5 or higher is required"));
        }
        let name = context.name().map_err(unavailable)?;
        let module = context
            .load_module(compile_kernel()?)
            .map_err(unavailable)?;
        let kernel = module
            .load_function("quantinfer_matmul")
            .map_err(unavailable)?;
        Ok(Self {
            stream: context.new_stream().map_err(unavailable)?,
            kernel,
            name,
            capabilities: projection_capabilities(),
            cpu: CpuScalar::new(),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    fn multiply(
        &self,
        lhs: &Tensor,
        rhs: ProjectionWeights<'_>,
        output_rows: bool,
    ) -> Result<Tensor> {
        let &[rows, inner] = lhs.shape().dims() else {
            return Err(invalid("left matrix must have rank 2"));
        };
        let &[rhs_inner, columns] = rhs.shape().dims() else {
            return Err(invalid("right matrix must have rank 2"));
        };
        if rows == 0 || inner == 0 || columns == 0 || inner != rhs_inner {
            return Err(invalid(
                "matrix dimensions must be nonzero and share an inner dimension",
            ));
        }
        if output_rows && !inner.is_multiple_of(32) {
            return Err(invalid(
                "packed output rows require an input width divisible by 32",
            ));
        }
        let rows = kernel_dimension(rows)?;
        let inner = kernel_dimension(inner)?;
        let columns = kernel_dimension(columns)?;
        kernel_elements(rows, inner)?;
        kernel_elements(inner, columns)?;
        let output_len = kernel_elements(rows, columns)? as usize;
        let column_tiles = columns.div_ceil(TILE);
        let grid = LaunchConfig {
            grid_dim: (column_tiles * rows.div_ceil(TILE), 1, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let encoding = rhs.encoding();
        let output_rows = u32::from(output_rows);
        let weights = rhs.bytes()?;
        let left = self.stream.clone_htod(lhs.as_f32()?).map_err(unavailable)?;
        let right = self
            .stream
            .clone_htod(weights.as_ref())
            .map_err(unavailable)?;
        let mut output = self
            .stream
            .alloc_zeros::<f32>(output_len)
            .map_err(unavailable)?;
        // The kernel ABI matches these arguments. Its guards and the checked dimensions
        // bound every read and write to the allocations owned by this stream.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(&left)
                .arg(&right)
                .arg(&mut output)
                .arg(&rows)
                .arg(&inner)
                .arg(&columns)
                .arg(&encoding)
                .arg(&output_rows)
                .arg(&column_tiles)
                .launch(grid)
        }
        .map_err(unavailable)?;
        let values = self.stream.clone_dtoh(&output).map_err(unavailable)?;
        Ok(Tensor::from_f32(
            vec![rows as usize, columns as usize],
            values,
        )?)
    }
}

impl RuntimeProvider for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

impl KernelBackend for CudaBackend {
    fn reuse_workspace(&self) -> bool {
        self.cpu.reuse_workspace()
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.multiply(lhs, ProjectionWeights::F32(rhs), false)
    }

    fn matmul_q8_0(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        self.multiply(lhs, ProjectionWeights::Q8(rhs), false)
    }

    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.multiply(lhs, ProjectionWeights::Q4(rhs), false)
    }

    fn matmul_q8_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor> {
        self.multiply(lhs, ProjectionWeights::Q8(rhs), true)
    }

    fn matmul_q4_0_output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.multiply(lhs, ProjectionWeights::Q4(rhs), true)
    }

    fn matmul_q4_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> Result<Tensor> {
        self.cpu.matmul_q4_k(lhs, rhs)
    }

    fn matmul_q5_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> Result<Tensor> {
        self.cpu.matmul_q5_k(lhs, rhs)
    }

    fn matmul_q6_k(&self, lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> Result<Tensor> {
        self.cpu.matmul_q6_k(lhs, rhs)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        self.cpu.rms_norm(input, weight, epsilon)
    }

    fn rms_norm_heads(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        self.cpu.rms_norm_heads(input, weight, epsilon)
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.cpu.softmax(input)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        self.cpu.silu_mul(gate, up)
    }

    fn apply_rope(&self, input: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Tensor> {
        self.cpu.apply_rope(input, cos, sin)
    }

    fn embedding(&self, table: &Tensor, token_ids: &[usize]) -> Result<Tensor> {
        self.cpu.embedding(table, token_ids)
    }
}

fn compile_kernel() -> Result<Ptx> {
    // The probe only loads and releases the installed NVRTC library.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(unavailable(
            "the CUDA NVRTC compiler library is unavailable",
        ));
    }
    compile_ptx_with_opts(
        KERNEL_SOURCE,
        CompileOptions {
            arch: Some("compute_75"),
            fmad: Some(false),
            name: Some("quantinfer_matmul.cu".to_owned()),
            ..Default::default()
        },
    )
    .map_err(unavailable)
}

fn kernel_dimension(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid("matrix dimension exceeds kernel indexing"))
}

fn kernel_elements(rows: u32, columns: u32) -> Result<u32> {
    rows.checked_mul(columns)
        .ok_or_else(|| invalid("matrix element count exceeds kernel indexing"))
}

fn invalid(message: &str) -> KernelError {
    KernelError::InvalidInput(format!("CUDA {message}"))
}

fn unavailable(error: impl std::fmt::Display) -> KernelError {
    KernelError::Unsupported(format!("CUDA {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the NVIDIA NVRTC compiler library"]
    fn compiles_matmul_to_ptx() {
        let ptx = compile_kernel().unwrap().to_src();
        assert!(ptx.contains(".entry quantinfer_matmul("));
        assert!(ptx.contains(".target sm_75"));
    }

    #[test]
    #[ignore = "requires a host without an NVIDIA driver"]
    fn reports_missing_driver() {
        let error = CudaBackend::open(0).unwrap_err();
        assert!(matches!(error, KernelError::Unsupported(_)));
        assert!(error.to_string().contains("driver library is unavailable"));
    }
}
