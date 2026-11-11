//! Optional Arm KleidiAI capability provider.
//!
//! The crate describes the runtime contract used by an external KleidiAI
//! kernel. It deliberately does not link a vendor library, allowing portable
//! builds to keep using the scalar or SIMD reference backend.

use std::collections::BTreeSet;

use quantinfer_core::{DType, Device, QuantizationScheme};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

#[derive(Debug, Clone)]
pub struct KleidiAiProvider {
    capabilities: Vec<KernelCapability>,
}

impl KleidiAiProvider {
    /// Build the Arm int8 capability contract.
    pub fn new() -> Self {
        Self {
            capabilities: vec![KernelCapability {
                device: Device::Cpu,
                dtype: DType::Q8_0,
                quantization: QuantizationScheme::Q8_0,
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

#[cfg(test)]
mod tests {
    use quantinfer_core::{DType, QuantizationScheme};
    use quantinfer_runtime::{ExecutionRequest, Operation, RuntimeProvider};

    use super::KleidiAiProvider;

    #[test]
    fn exposes_explicit_q8_capability() {
        let mut request = ExecutionRequest::cpu(DType::Q8_0, QuantizationScheme::Q8_0);
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(32);
        let plan = KleidiAiProvider::new().resolve(&request).unwrap();
        assert_eq!(plan.provider_name, "kleidiai");
        assert_eq!(plan.kernel_provider_name, "kleidiai");
    }
}
