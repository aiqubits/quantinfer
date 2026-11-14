//! CPU ISA, runtime configuration, and shared tensor validation.

use quantinfer_core::Tensor;
#[cfg(feature = "simd")]
use quantinfer_core::{Q4_0Block, Q8_0Block};
use quantinfer_kernel::{KernelError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuIsa {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl CpuIsa {
    pub fn detect() -> Self {
        #[cfg(feature = "simd")]
        for isa in [Self::Avx512, Self::Avx2, Self::Neon] {
            if isa.scaled_add().is_some() {
                return isa;
            }
        }
        Self::Scalar
    }

    #[cfg(feature = "simd")]
    fn scaled_add(self) -> Option<quantinfer_backend_cpu_simd::ScaledAdd> {
        use quantinfer_backend_cpu_simd::{ScaledAdd, SimdIsa};

        let isa = match self {
            Self::Scalar => return None,
            Self::Avx2 => SimdIsa::Avx2,
            Self::Avx512 => SimdIsa::Avx512,
            Self::Neon => SimdIsa::Neon,
        };
        ScaledAdd::for_isa(isa)
    }

    #[cfg(feature = "simd")]
    fn quantized_dot(self) -> Option<quantinfer_backend_cpu_simd::QuantizedDot> {
        use quantinfer_backend_cpu_simd::{QuantizedDot, SimdIsa};

        let isa = match self {
            Self::Scalar => return None,
            Self::Avx2 => SimdIsa::Avx2,
            Self::Avx512 => SimdIsa::Avx512,
            Self::Neon => SimdIsa::Neon,
        };
        QuantizedDot::for_isa(isa)
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
    pub(crate) config: CpuRuntimeConfig,
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

    pub(crate) fn f32_values<'a>(&self, tensor: &'a Tensor, name: &str) -> Result<&'a [f32]> {
        tensor
            .as_f32()
            .map_err(|error| KernelError::InvalidInput(format!("{name} must be F32: {error}")))
    }

    pub(crate) fn expect_matrix(&self, tensor: &Tensor, name: &str) -> Result<(usize, usize)> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(KernelError::InvalidInput(format!(
                "{name} must be rank 2, got rank {}",
                dims.len()
            )));
        }
        Ok((dims[0], dims[1]))
    }

    pub(crate) fn expect_last_dim(&self, tensor: &Tensor, name: &str) -> Result<usize> {
        tensor.shape().last_dim().ok_or_else(|| {
            KernelError::InvalidInput(format!("{name} must have at least one dimension"))
        })
    }
}
