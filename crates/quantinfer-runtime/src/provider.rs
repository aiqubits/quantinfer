use std::collections::BTreeSet;
use std::fmt;

use quantinfer_core::{DType, Device, QuantizationScheme};

use crate::{Result, RuntimeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operation {
    Matmul,
    QuantizedMatmul,
    Attention,
    RmsNorm,
    Activation,
    RoPE,
    Embedding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCapability {
    pub device: Device,
    pub dtype: DType,
    pub quantization: QuantizationScheme,
    pub tensor_layout: String,
    pub supported_operations: BTreeSet<Operation>,
    pub alignment: usize,
    pub head_dimension: Option<usize>,
    pub maximum_sequence_length: Option<usize>,
}

impl KernelCapability {
    pub fn supports(&self, request: &ExecutionRequest) -> bool {
        self.device == request.device
            && self.dtype == request.dtype
            && self.quantization == request.quantization
            && (request.tensor_layout.is_empty() || self.tensor_layout == request.tensor_layout)
            && request
                .operation
                .is_none_or(|operation| self.supported_operations.contains(&operation))
            && request
                .alignment
                .is_none_or(|alignment| alignment % self.alignment == 0)
            && request
                .head_dimension
                .is_none_or(|dimension| self.head_dimension.is_none_or(|max| dimension <= max))
            && request
                .maximum_sequence_length
                .is_none_or(|length| self.maximum_sequence_length.is_none_or(|max| length <= max))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionRequest {
    pub device: Device,
    pub dtype: DType,
    pub quantization: QuantizationScheme,
    pub tensor_layout: String,
    pub operation: Option<Operation>,
    pub alignment: Option<usize>,
    pub head_dimension: Option<usize>,
    pub maximum_sequence_length: Option<usize>,
}

impl ExecutionRequest {
    pub fn cpu(dtype: DType, quantization: QuantizationScheme) -> Self {
        Self {
            device: Device::Cpu,
            dtype,
            quantization,
            tensor_layout: String::new(),
            operation: None,
            alignment: None,
            head_dimension: None,
            maximum_sequence_length: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub provider_name: String,
    pub kernel_name: String,
    pub capability: KernelCapability,
}

pub trait RuntimeProvider: fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &[KernelCapability];

    fn resolve(&self, request: &ExecutionRequest) -> Result<ExecutionPlan> {
        let capability = self
            .capabilities()
            .iter()
            .find(|capability| capability.supports(request))
            .cloned()
            .ok_or_else(|| {
                RuntimeError::InvalidInput(format!(
                    "provider {} has no kernel capability for {:?}",
                    self.name(),
                    request
                ))
            })?;
        Ok(ExecutionPlan {
            provider_name: self.name().to_owned(),
            kernel_name: format!("{}::{:?}", self.name(), capability.operation_name()),
            capability,
        })
    }
}

pub trait ExecutionBackend: fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &[KernelCapability];
}

impl<T: RuntimeProvider + ?Sized> ExecutionBackend for T {
    fn name(&self) -> &str {
        RuntimeProvider::name(self)
    }

    fn capabilities(&self) -> &[KernelCapability] {
        RuntimeProvider::capabilities(self)
    }
}

impl KernelCapability {
    fn operation_name(&self) -> &'static str {
        if self.quantization != QuantizationScheme::None {
            "quantized"
        } else {
            "reference"
        }
    }
}

#[derive(Debug, Default)]
pub struct KernelRegistry {
    providers: Vec<Box<dyn RuntimeProvider>>,
}

pub type RuntimeProviderRegistry = KernelRegistry;

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<P>(&mut self, provider: P)
    where
        P: RuntimeProvider + 'static,
    {
        self.providers.push(Box::new(provider));
    }

    pub fn providers(&self) -> impl Iterator<Item = &dyn RuntimeProvider> {
        self.providers.iter().map(Box::as_ref)
    }

    pub fn resolve(&self, request: &ExecutionRequest) -> Result<ExecutionPlan> {
        self.providers
            .iter()
            .find_map(|provider| provider.resolve(request).ok())
            .ok_or_else(|| {
                RuntimeError::InvalidInput(format!(
                    "no registered runtime provider supports {:?}",
                    request
                ))
            })
    }
}

#[derive(Debug, Default)]
pub struct ExecutionResolver {
    registry: KernelRegistry,
}

impl ExecutionResolver {
    pub fn new(registry: KernelRegistry) -> Self {
        Self { registry }
    }

    pub fn with_default_cpu() -> Self {
        Self::new(default_registry())
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }

    pub fn resolve(&self, request: &ExecutionRequest) -> Result<ExecutionPlan> {
        self.registry.resolve(request)
    }
}

#[derive(Debug, Clone)]
pub struct CpuScalarProvider {
    capabilities: Vec<KernelCapability>,
}

impl Default for CpuScalarProvider {
    fn default() -> Self {
        let operations: BTreeSet<Operation> = [
            Operation::Matmul,
            Operation::QuantizedMatmul,
            Operation::Attention,
            Operation::RmsNorm,
            Operation::Activation,
            Operation::RoPE,
            Operation::Embedding,
        ]
        .into_iter()
        .collect();
        let base = |dtype, quantization, alignment| KernelCapability {
            device: Device::Cpu,
            dtype,
            quantization,
            tensor_layout: "row_major".to_owned(),
            supported_operations: operations.clone(),
            alignment,
            head_dimension: None,
            maximum_sequence_length: None,
        };
        Self {
            capabilities: vec![
                base(DType::F32, QuantizationScheme::None, 1),
                base(DType::Q8_0, QuantizationScheme::Q8_0, 32),
                base(DType::Q4_0, QuantizationScheme::Q4_0, 32),
                base(DType::Q4_K, QuantizationScheme::Q4_K, 256),
            ],
        }
    }
}

impl RuntimeProvider for CpuScalarProvider {
    fn name(&self) -> &str {
        "cpu-scalar"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

pub fn default_registry() -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    registry.register(CpuScalarProvider::default());
    registry
}

#[cfg(test)]
mod tests {
    use quantinfer_core::{DType, QuantizationScheme};

    use super::{default_registry, ExecutionRequest, Operation};

    #[test]
    fn resolves_cpu_scalar_capabilities_for_q4() {
        let mut request = ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0);
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(32);
        let plan = default_registry().resolve(&request).unwrap();
        assert_eq!(plan.provider_name, "cpu-scalar");
        assert_eq!(plan.capability.dtype, DType::Q4_0);
    }

    #[test]
    fn rejects_unsupported_device_or_layout() {
        let mut request = ExecutionRequest::cpu(DType::F32, QuantizationScheme::None);
        request.tensor_layout = "blocked".to_owned();
        assert!(default_registry().resolve(&request).is_err());
    }
}
