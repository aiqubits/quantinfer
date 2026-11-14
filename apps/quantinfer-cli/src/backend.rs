use quantinfer::cpu::{CpuModel, DenseModel};
use quantinfer::kernel::KernelBackend;
use quantinfer_format::{GgufReader, ModelManifest};
use quantinfer_runtime::{
    DecoderModel, InferenceProfile, KvCache, LoraAdapter, Result as RuntimeResult,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu,
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
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
            "cuda" => {
                #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
                {
                    Ok(Self::Cuda)
                }
                #[cfg(not(all(
                    feature = "cuda",
                    any(target_os = "linux", target_os = "windows")
                )))]
                {
                    Err("CUDA generation requires a Linux or Windows CLI built with --features cuda".to_owned())
                }
            }
            "metal" => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(Self::Metal)
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    Err(
                        "Metal generation requires a macOS CLI built with --features metal"
                            .to_owned(),
                    )
                }
            }
            name => Err(format!(
                "unknown --backend {name}; use cpu, wgpu, cuda, or metal"
            )),
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
            Self::Wgpu => load_gpu(reader, manifest, || {
                quantinfer::wgpu::WgpuBackend::open(false)
            }),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Self::Cuda => load_gpu(reader, manifest, || quantinfer::cuda::CudaBackend::open(0)),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal => load_gpu(reader, manifest, quantinfer::metal::MetalBackend::open),
        }
    }
}

#[cfg(any(
    all(feature = "wgpu", not(target_arch = "wasm32")),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "metal", target_os = "macos")
))]
fn load_gpu<B: KernelBackend + quantinfer_runtime::RuntimeProvider + 'static>(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
    open: impl FnOnce() -> quantinfer::kernel::Result<B>,
) -> Result<Box<dyn Generator>, String> {
    use quantinfer_core::{DType, Device, QuantizationScheme, OUTPUT_ROWS_LAYOUT};
    use quantinfer_format::GgmlType;
    use quantinfer_format::GGML_OUTPUT_ROWS_LAYOUT;
    use quantinfer_runtime::{ExecutionRequest, Operation};

    let manifest = manifest.or(reader.embedded_manifest());
    let ggml = reader
        .uses_ggml_output_rows()
        .map_err(|error| error.to_string())?;
    let backend = open().map_err(|error| error.to_string())?;
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
                "model package {} has no {} execution variant",
                manifest.model_id, plan.provider_name
            ));
        }
        if manifest.tensors.iter().any(|tensor| {
            let layout = tensor.descriptor.physical_layout.as_str();
            layout != "row_major" && !(ggml && layout == GGML_OUTPUT_ROWS_LAYOUT)
        }) {
            return Err(format!(
                "model package declares an unsupported {} tensor layout",
                plan.provider_name
            ));
        }
    }
    for (kind, dtype, quantization) in [
        (GgmlType::Q8_0, DType::Q8_0, QuantizationScheme::Q8_0),
        (GgmlType::Q4_0, DType::Q4_0, QuantizationScheme::Q4_0),
    ] {
        if reader.tensors().iter().any(|tensor| tensor.dtype == kind) {
            let tensor_layout = if ggml {
                OUTPUT_ROWS_LAYOUT
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
    fn attach_lora(&mut self, adapter: LoraAdapter) -> RuntimeResult<()>;
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

    fn attach_lora(&mut self, adapter: LoraAdapter) -> RuntimeResult<()> {
        self.attach_lora(adapter)
    }
}
