mod executor;
mod ffi;
mod resources;
mod runtime;
pub use executor::CannMatmulExecutor;
pub use runtime::{AscendMatmulPlan, AscendRuntime, AscendRuntimeError};

use std::collections::BTreeSet;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor,
};
use quantinfer_kernel::{KernelBackend, Result as KernelResult};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

/// F32 capability description for an explicitly selected Ascend executor.
#[derive(Debug, Clone)]
pub struct AscendProvider {
    capabilities: Vec<KernelCapability>,
}

impl RuntimeProvider for AscendProvider {
    fn name(&self) -> &str {
        "ascend"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

/// The execution hook implemented by `CannMatmulExecutor` or an external ACL
/// Tensor adapter. The executor owns
/// tensor creation, device memory and stream synchronization; this crate keeps
/// the model/runtime boundary independent of those CANN details.
pub trait AscendMatmulExecutor: std::fmt::Debug + Send + Sync {
    fn matmul_f32(&self, lhs: &Tensor, rhs: &Tensor) -> KernelResult<Tensor>;
}

/// Runtime backend for an Ascend executor with explicit CPU fallbacks for
/// operators not yet represented by the ACLNN adapter.
#[derive(Debug)]
pub struct AscendBackend<E> {
    executor: E,
    fallback: CpuScalar,
    capabilities: Vec<KernelCapability>,
}

impl<E: AscendMatmulExecutor> AscendBackend<E> {
    pub fn new(executor: E) -> Self {
        let provider = AscendProvider {
            capabilities: vec![float_matmul_capability()],
        };
        Self {
            executor,
            fallback: CpuScalar::new(),
            capabilities: provider.capabilities.clone(),
        }
    }
}

impl AscendBackend<CannMatmulExecutor> {
    pub fn open(device: i32) -> KernelResult<Self> {
        CannMatmulExecutor::open(device).map(Self::new)
    }
}

impl<E: AscendMatmulExecutor> RuntimeProvider for AscendBackend<E> {
    fn name(&self) -> &str {
        "ascend"
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

impl<E: AscendMatmulExecutor> KernelBackend for AscendBackend<E> {
    fn backend_id(&self) -> Option<quantinfer_kernel::BackendId> {
        Some(quantinfer_kernel::BackendId::Ascend)
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> KernelResult<Tensor> {
        self.executor.matmul_f32(lhs, rhs)
    }

    fallback_kernel! {
        reuse_workspace() -> bool,
        matmul_q8_0(lhs: &Tensor, rhs: &QuantizedTensor) -> KernelResult<Tensor>,
        matmul_q4_0(lhs: &Tensor, rhs: &QuantizedTensor4) -> KernelResult<Tensor>,
        matmul_q8_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor) -> KernelResult<Tensor>,
        matmul_q4_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor4) -> KernelResult<Tensor>,
        matmul_q4_k(lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> KernelResult<Tensor>,
        matmul_q5_k(lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> KernelResult<Tensor>,
        matmul_q6_k(lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> KernelResult<Tensor>,
        rms_norm(input: &Tensor, weight: &Tensor, epsilon: f32) -> KernelResult<Tensor>,
        rms_norm_heads(input: &Tensor, weight: &Tensor, epsilon: f32) -> KernelResult<Tensor>,
        softmax(input: &Tensor) -> KernelResult<Tensor>,
        silu_mul(gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>,
        apply_rope(input: &Tensor, cos: &[f32], sin: &[f32]) -> KernelResult<Tensor>,
        embedding(table: &Tensor, token_ids: &[usize]) -> KernelResult<Tensor>,
    }
}

fn float_matmul_capability() -> KernelCapability {
    KernelCapability {
        device: Device::Npu,
        dtype: DType::F32,
        quantization: QuantizationScheme::None,
        tensor_layout: "row_major".to_owned(),
        supported_operations: BTreeSet::from([Operation::Matmul]),
        alignment: 1,
        head_dimension: None,
        maximum_sequence_length: None,
    }
}

#[cfg(test)]
mod tests {
    use quantinfer_core::Tensor;
    use quantinfer_kernel::KernelBackend;
    use quantinfer_runtime::RuntimeProvider;

    use super::{
        AscendBackend, AscendMatmulExecutor, AscendRuntime, AscendRuntimeError, CpuScalar,
    };

    #[test]
    fn missing_library_is_reported_without_panicking() {
        std::env::set_var("ASCEND_ACL_LIBRARY", "/missing/libascendcl.so");
        let result = AscendRuntime::open();
        assert!(matches!(
            result,
            Err(AscendRuntimeError::LibraryUnavailable(_))
        ));
    }

    #[test]
    fn operator_library_is_not_assumed_when_acl_is_missing() {
        std::env::set_var("ASCEND_ACL_LIBRARY", "/missing/libascendcl.so");
        std::env::set_var("ASCEND_OP_LIBRARY", "/missing/libopapi.so");
        assert!(AscendRuntime::open().is_err());
    }

    #[derive(Debug)]
    struct ReferenceExecutor;

    impl AscendMatmulExecutor for ReferenceExecutor {
        fn matmul_f32(&self, lhs: &Tensor, rhs: &Tensor) -> quantinfer_kernel::Result<Tensor> {
            CpuScalar::new().matmul(lhs, rhs)
        }
    }

    #[test]
    fn runtime_backend_routes_float_matmul_to_executor() {
        let backend = AscendBackend::new(ReferenceExecutor);
        let lhs = Tensor::from_f32(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let rhs = Tensor::from_f32(vec![2, 1], vec![4.0, 5.0]).unwrap();
        let output = backend.matmul(&lhs, &rhs).unwrap();
        assert_eq!(output.as_f32().unwrap(), &[23.0]);
        assert_eq!(backend.name(), "ascend");
    }
}
