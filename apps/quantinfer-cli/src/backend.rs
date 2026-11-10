use quantinfer::cpu::{CpuModel, DenseModel};
use quantinfer::kernel::KernelBackend;
use quantinfer_format::{GgufReader, ModelManifest};
use quantinfer_runtime::{DecoderModel, InferenceProfile, KvCache, Result as RuntimeResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu,
}

impl Backend {
    pub fn parse(name: Option<&str>) -> Result<Self, String> {
        match name.unwrap_or("cpu") {
            "cpu" => Ok(Self::Cpu),
            "wgpu" => {
                #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
                {
                    Ok(Self::Wgpu)
                }
                #[cfg(not(all(feature = "wgpu", not(target_arch = "wasm32"))))]
                {
                    Err(
                        "WGPU generation requires a native CLI built with --features wgpu"
                            .to_owned(),
                    )
                }
            }
            name => Err(format!("unknown --backend {name}; use cpu or wgpu")),
        }
    }

    pub fn load(
        self,
        reader: &GgufReader,
        manifest: Option<&ModelManifest>,
        profile: Option<&InferenceProfile>,
    ) -> Result<Box<dyn Generator>, String> {
        match self {
            Self::Cpu => CpuModel::from_gguf(reader, manifest, profile)
                .map(|model| Box::new(model) as Box<dyn Generator>)
                .map_err(|error| error.to_string()),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Self::Wgpu => load_wgpu(reader, manifest),
        }
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
fn load_wgpu(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
) -> Result<Box<dyn Generator>, String> {
    use quantinfer::cpu::model_architecture;
    use quantinfer::wgpu::WgpuBackend;
    use quantinfer_core::{DType, Device, QuantizationScheme, GGML_OUTPUT_ROWS_LAYOUT};
    use quantinfer_format::GgmlType;
    use quantinfer_runtime::{ExecutionRequest, Operation, RuntimeProvider};

    let manifest = manifest.or(reader.embedded_manifest());
    let architecture = model_architecture(reader, manifest).map_err(|error| error.to_string())?;
    let ggml = architecture == "llama"
        && reader
            .llama_uses_ggml_layout()
            .map_err(|error| error.to_string())?;
    let backend = WgpuBackend::open(false).map_err(|error| error.to_string())?;
    let request = ExecutionRequest {
        device: Device::Gpu,
        tensor_layout: "row_major".to_owned(),
        operation: Some(Operation::Matmul),
        alignment: manifest.and_then(|manifest| manifest.kernel_requirements.alignment),
        ..ExecutionRequest::cpu(DType::F32, QuantizationScheme::None)
    };
    let plan = backend
        .resolve(&request)
        .map_err(|error| error.to_string())?;
    if let Some(manifest) = manifest {
        if !manifest.execution_variants.is_empty()
            && !manifest.execution_variants.iter().any(|variant| {
                variant.runtime_provider == plan.provider_name
                    && variant.kernel_provider == plan.kernel_provider_name
                    && variant.quantization == manifest.quantization
            })
        {
            return Err(format!(
                "model package {} has no WGPU execution variant",
                manifest.model_id
            ));
        }
        if manifest.tensors.iter().any(|tensor| {
            let layout = tensor.descriptor.physical_layout.as_str();
            layout != "row_major" && !(ggml && layout == GGML_OUTPUT_ROWS_LAYOUT)
        }) {
            return Err("model package declares an unsupported WGPU tensor layout".to_owned());
        }
    }
    for (kind, dtype, quantization) in [
        (GgmlType::Q8_0, DType::Q8_0, QuantizationScheme::Q8_0),
        (GgmlType::Q4_0, DType::Q4_0, QuantizationScheme::Q4_0),
    ] {
        if reader.tensors().iter().any(|tensor| tensor.dtype == kind) {
            let tensor_layout = if ggml {
                GGML_OUTPUT_ROWS_LAYOUT
            } else {
                "row_major"
            };
            backend
                .resolve(&ExecutionRequest {
                    dtype,
                    quantization,
                    operation: Some(Operation::QuantizedMatmul),
                    tensor_layout: tensor_layout.to_owned(),
                    alignment: Some(32),
                    ..request
                })
                .map_err(|error| error.to_string())?;
        }
    }
    DenseModel::with_backend(backend, reader, manifest)
        .map(|model| Box::new(model) as Box<dyn Generator>)
        .map_err(|error| error.to_string())
}

pub trait Generator: DecoderModel {
    fn vocab_size(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn new_cache(&self, context_length: usize) -> RuntimeResult<KvCache>;
}

impl<B: KernelBackend> Generator for DenseModel<B> {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }

    fn new_cache(&self, context_length: usize) -> RuntimeResult<KvCache> {
        self.new_cache(context_length)
    }
}
