//! Optional Arm KleidiAI capability provider.
//!
//! The crate describes the runtime contract used by an external KleidiAI
//! kernel. It deliberately does not link a vendor library, allowing portable
//! builds to keep using the scalar or SIMD reference backend.

use std::collections::BTreeSet;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor,
};
use quantinfer_kernel::{KernelBackend, Result};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

#[derive(Debug, Clone)]
pub struct KleidiAiProvider {
    capabilities: Vec<KernelCapability>,
}

impl KleidiAiProvider {
    /// Build the Arm int4 capability contract used by the KleidiAI matmul
    /// micro-kernels (f32 activations multiplied by packed int4 weights).
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

    /// Return a provider only on AArch64, where KleidiAI can be linked.
    /// Registration is explicit so an application can choose its linked
    /// KleidiAI build and retain a scalar fallback when it is absent.
    pub fn try_new() -> Option<Self> {
        Self::is_supported_target().then(Self::new)
    }

    pub const fn is_supported_target() -> bool {
        cfg!(target_arch = "aarch64")
    }
}

impl Default for KleidiAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeProvider for KleidiAiProvider {
    fn name(&self) -> &str {
        "kleidiai"
    }

    fn kernel_provider_name(&self) -> &str {
        "kleidiai"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

/// The execution hook a linked KleidiAI build must implement. Packing and
/// thread scheduling remain the responsibility of the embedding application.
pub trait KleidiAiKernel: std::fmt::Debug + Send + Sync {
    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor>;
}

#[derive(Debug)]
pub struct KleidiAiBackend<K> {
    kernel: K,
    fallback: CpuScalar,
    capabilities: Vec<KernelCapability>,
}

impl<K: KleidiAiKernel> KleidiAiBackend<K> {
    pub fn new(kernel: K) -> Self {
        let provider = KleidiAiProvider::new();
        Self {
            kernel,
            fallback: CpuScalar::new(),
            capabilities: provider.capabilities().to_vec(),
        }
    }
}

impl<K: KleidiAiKernel> RuntimeProvider for KleidiAiBackend<K> {
    fn name(&self) -> &str {
        "kleidiai"
    }

    fn kernel_provider_name(&self) -> &str {
        "kleidiai"
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

impl<K: KleidiAiKernel> KernelBackend for KleidiAiBackend<K> {
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

    use super::{KleidiAiBackend, KleidiAiKernel, KleidiAiProvider};

    #[test]
    fn exposes_explicit_q4_capability() {
        let mut request = ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0);
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(32);
        let plan = KleidiAiProvider::new().resolve(&request).unwrap();
        assert_eq!(plan.provider_name, "kleidiai");
        assert_eq!(plan.kernel_provider_name, "kleidiai");
    }

    #[derive(Debug)]
    struct ReferenceKernel;

    impl KleidiAiKernel for ReferenceKernel {
        fn matmul_q4_0(
            &self,
            lhs: &Tensor,
            rhs: &QuantizedTensor4,
        ) -> quantinfer_kernel::Result<Tensor> {
            CpuScalar::new().matmul_q4_0(lhs, rhs)
        }
    }

    #[test]
    fn routes_int4_projection_through_the_linked_kernel() {
        let backend = KleidiAiBackend::new(ReferenceKernel);
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
