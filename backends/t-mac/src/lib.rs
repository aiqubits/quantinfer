//! Optional T-MAC capability provider.
//!
//! T-MAC is kept behind a small planner crate so a vendor or downstream
//! implementation can supply its lookup-table kernel without adding a native
//! dependency to the core runtime. The provider only advertises a capability
//! when the caller explicitly registers it.

use std::collections::BTreeSet;

use quantinfer_core::{DType, Device, QuantizationScheme};
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

#[cfg(test)]
mod tests {
    use quantinfer_core::{DType, QuantizationScheme};
    use quantinfer_runtime::{ExecutionRequest, Operation, RuntimeProvider};

    use super::TmacProvider;

    #[test]
    fn exposes_explicit_q4_capability() {
        let mut request = ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0);
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(32);
        let plan = TmacProvider::new().resolve(&request).unwrap();
        assert_eq!(plan.provider_name, "t-mac");
        assert_eq!(plan.kernel_provider_name, "t-mac");
    }
}
