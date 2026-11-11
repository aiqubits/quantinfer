//! Optional T-MAC capability provider.
//!
//! T-MAC is kept behind a small planner crate so a vendor or downstream
//! implementation can supply its lookup-table kernel without adding a native
//! dependency to the core runtime. The provider only advertises a capability
//! when the caller explicitly registers it.

use std::collections::BTreeSet;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor,
};
use quantinfer_kernel::{KernelBackend, Result};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

#[derive(Debug, Clone)]
pub struct TmacProvider {
    capabilities: Vec<KernelCapability>,
}

impl TmacProvider {
    /// Build the standard T-MAC planner contract for packed 4-bit matmul.
    pub fn new() -> Self {
        Self {
            capabilities: vec![KernelCapability {
                device: Device::Cpu,
                dtype: DType::Q4_0,
                quantization: QuantizationScheme::Q4_0,
                tensor_layout: "row_major".to_owned(),
                supported_operations: BTreeSet::from([Operation::QuantizedMatmul]),
                alignment: 32,
                head_dimension: None,
                maximum_sequence_length: None,
            }],
        }
    }

    /// Return a provider only on targets supported by the portable T-MAC ABI.
    /// Registration remains explicit so applications can pair this contract
    /// with their linked or dynamically loaded T-MAC kernel.
    pub fn try_new() -> Option<Self> {
        Self::is_supported_target().then(Self::new)
    }

    pub const fn is_supported_target() -> bool {
        cfg!(any(target_arch = "aarch64", target_arch = "x86_64"))
    }
}

impl Default for TmacProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeProvider for TmacProvider {
    fn name(&self) -> &str {
        "t-mac"
    }

    fn kernel_provider_name(&self) -> &str {
        "t-mac"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

/// The small execution hook a linked T-MAC build must implement.
pub trait TmacKernel: std::fmt::Debug + Send + Sync {
    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor>;
}

/// T-MAC backend with a caller-supplied LUT kernel and safe CPU fallbacks for
/// operators that are not part of the low-bit projection path.
#[derive(Debug)]
pub struct TmacBackend<K> {
    kernel: K,
    fallback: CpuScalar,
    capabilities: Vec<KernelCapability>,
}

impl<K: TmacKernel> TmacBackend<K> {
    pub fn new(kernel: K) -> Self {
        let provider = TmacProvider::new();
        Self {
            kernel,
            fallback: CpuScalar::new(),
            capabilities: provider.capabilities().to_vec(),
        }
    }
}

impl<K: TmacKernel> RuntimeProvider for TmacBackend<K> {
    fn name(&self) -> &str {
        "t-mac"
    }

    fn kernel_provider_name(&self) -> &str {
        "t-mac"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

macro_rules! fallback_kernel {
    ($($method:ident($($arg:ident : $ty:ty),*) -> $ret:ty),* $(,)?) => {
        $(fn $method(&self, $($arg: $ty),*) -> $ret {
            self.fallback.$method($($arg),*)
        })*
    };
}

impl<K: TmacKernel> KernelBackend for TmacBackend<K> {
    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.kernel.matmul_q4_0(lhs, rhs)
    }

    fallback_kernel! {
        reuse_workspace() -> bool,
        matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>,
        matmul_q8_0(lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor>,
        matmul_q8_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor>,
        matmul_q4_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor>,
        matmul_q4_k(lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> Result<Tensor>,
        matmul_q5_k(lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> Result<Tensor>,
        matmul_q6_k(lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> Result<Tensor>,
        rms_norm(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor>,
        rms_norm_heads(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor>,
        softmax(input: &Tensor) -> Result<Tensor>,
        silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor>,
        apply_rope(input: &Tensor, cos: &[f32], sin: &[f32]) -> Result<Tensor>,
        embedding(table: &Tensor, token_ids: &[usize]) -> Result<Tensor>,
    }
}

#[cfg(test)]
mod tests {
    use quantinfer_core::{DType, QuantizationScheme};
    use quantinfer_runtime::{ExecutionRequest, Operation, RuntimeProvider};

    use quantinfer_backend_cpu::CpuScalar;
    use quantinfer_core::{Q4_0Block, QuantizedTensor4, Tensor};
    use quantinfer_kernel::KernelBackend;

    use super::{TmacBackend, TmacKernel, TmacProvider};

    #[test]
    fn exposes_explicit_q4_capability() {
        let mut request = ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0);
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(32);
        let plan = TmacProvider::new().resolve(&request).unwrap();
        assert_eq!(plan.provider_name, "t-mac");
        assert_eq!(plan.kernel_provider_name, "t-mac");
    }

    #[derive(Debug)]
    struct ReferenceKernel;

    impl TmacKernel for ReferenceKernel {
        fn matmul_q4_0(
            &self,
            lhs: &Tensor,
            rhs: &QuantizedTensor4,
        ) -> quantinfer_kernel::Result<Tensor> {
            CpuScalar::new().matmul_q4_0(lhs, rhs)
        }
    }

    #[test]
    fn routes_q4_projection_through_the_linked_kernel() {
        let backend = TmacBackend::new(ReferenceKernel);
        let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
        let rhs = QuantizedTensor4::from_blocks(
            vec![32, 1],
            vec![Q4_0Block {
                scale: 1.0,
                values: [8; 32],
            }],
        )
        .unwrap();
        let output = backend.matmul_q4_0(&lhs, &rhs).unwrap();
        assert_eq!(output.as_f32().unwrap(), &[0.0]);
    }
}
