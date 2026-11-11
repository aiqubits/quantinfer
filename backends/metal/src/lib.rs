//! Native Metal compute provider built on the shared WGPU shader path.

#![forbid(unsafe_code)]

#[cfg(all(feature = "native", target_os = "macos"))]
mod native {
    use quantinfer_backend_wgpu::WgpuBackend;
    use quantinfer_core::{
        QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K,
        QuantizedTensorQ6K, Tensor,
    };
    use quantinfer_kernel::{KernelBackend, Result};
    use quantinfer_runtime::{KernelCapability, RuntimeProvider};

    /// Tiled F32, Q4_0, and Q8_0 projections on Apple's Metal adapter.
    #[derive(Debug, Clone)]
    pub struct MetalBackend(WgpuBackend);

    impl MetalBackend {
        pub fn open() -> Result<Self> {
            WgpuBackend::open_metal().map(Self)
        }

        pub fn adapter_name(&self) -> &str {
            self.0.adapter_name()
        }
    }

    impl RuntimeProvider for MetalBackend {
        fn name(&self) -> &str {
            "metal"
        }

        fn capabilities(&self) -> &[KernelCapability] {
            self.0.capabilities()
        }
    }

    macro_rules! delegate {
        ($($method:ident($($arg:ident : $ty:ty),*) -> $ret:ty),* $(,)?) => {
            $(fn $method(&self, $($arg: $ty),*) -> $ret {
                self.0.$method($($arg),*)
            })*
        };
    }

    impl KernelBackend for MetalBackend {
        delegate! {
            reuse_workspace() -> bool,
            matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>,
            matmul_q8_0(lhs: &Tensor, rhs: &QuantizedTensor) -> Result<Tensor>,
            matmul_q4_0(lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor>,
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
        use super::MetalBackend;

        #[test]
        #[ignore = "requires an Apple Metal adapter"]
        fn opens_metal_adapter() {
            let backend = MetalBackend::open().unwrap();
            assert!(!backend.adapter_name().is_empty());
            assert_eq!(backend.name(), "metal");
        }
    }
}

#[cfg(all(feature = "native", target_os = "macos"))]
pub use native::MetalBackend;
