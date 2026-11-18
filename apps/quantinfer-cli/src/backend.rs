#[cfg(any(
    test,
    feature = "tmac",
    all(feature = "wgpu", not(target_arch = "wasm32")),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "metal", target_os = "macos"),
    all(feature = "ascend", target_os = "linux"),
    all(
        feature = "kleidiai",
        target_arch = "aarch64",
        target_os = "linux",
        target_endian = "little"
    )
))]
mod gpu_plan;

use quantinfer::cpu::{CpuModel, DenseModel};
use quantinfer::kernel::KernelBackend;
use quantinfer_format::{GgufReader, ModelManifest};
use quantinfer_runtime::{
    DecoderModel, InferenceProfile, KvCache, LoraAdapter, Result as RuntimeResult,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "tmac")]
    Tmac,
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu,
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
    #[cfg(all(feature = "ascend", target_os = "linux"))]
    Ascend,
    #[cfg(all(
        feature = "kleidiai",
        target_arch = "aarch64",
        target_os = "linux",
        target_endian = "little"
    ))]
    KleidiAi,
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
            "ascend" => {
                #[cfg(all(feature = "ascend", target_os = "linux"))]
                {
                    Ok(Self::Ascend)
                }
                #[cfg(not(all(feature = "ascend", target_os = "linux")))]
                {
                    Err(
                        "Ascend generation requires a Linux CLI built with --features ascend"
                            .into(),
                    )
                }
            }
            "kleidiai" => {
                #[cfg(all(
                    feature = "kleidiai",
                    target_arch = "aarch64",
                    target_os = "linux",
                    target_endian = "little"
                ))]
                {
                    Ok(Self::KleidiAi)
                }
                #[cfg(not(all(
                    feature = "kleidiai",
                    target_arch = "aarch64",
                    target_os = "linux",
                    target_endian = "little"
                )))]
                {
                    Err(
                        "KleidiAI requires a Linux AArch64 CLI built with --features kleidiai"
                            .into(),
                    )
                }
            }
            "t-mac" => {
                #[cfg(feature = "tmac")]
                {
                    Ok(Self::Tmac)
                }
                #[cfg(not(feature = "tmac"))]
                {
                    Err("T-MAC LUT4 requires a CLI built with --features tmac".into())
                }
            }
            name => Err(format!(
                "unknown --backend {name}; use cpu, wgpu, cuda, metal, ascend, kleidiai, or t-mac"
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
            #[cfg(feature = "tmac")]
            Self::Tmac => load_accelerator(
                reader,
                manifest,
                quantinfer::tmac::TmacBackend::open,
                quantinfer::core::Device::Cpu,
            ),
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
            #[cfg(all(feature = "ascend", target_os = "linux"))]
            Self::Ascend => load_accelerator(
                reader,
                manifest,
                || quantinfer::ascend::AscendBackend::open(0),
                quantinfer::core::Device::Npu,
            ),
            #[cfg(all(
                feature = "kleidiai",
                target_arch = "aarch64",
                target_os = "linux",
                target_endian = "little"
            ))]
            Self::KleidiAi => load_accelerator(
                reader,
                manifest,
                quantinfer::kleidiai::KleidiAiBackend::open,
                quantinfer::core::Device::Cpu,
            ),
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
    load_accelerator(reader, manifest, open, quantinfer::core::Device::Gpu)
}

#[cfg(any(
    feature = "tmac",
    all(feature = "wgpu", not(target_arch = "wasm32")),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "metal", target_os = "macos"),
    all(feature = "ascend", target_os = "linux"),
    all(
        feature = "kleidiai",
        target_arch = "aarch64",
        target_os = "linux",
        target_endian = "little"
    )
))]
fn load_accelerator<B: KernelBackend + quantinfer_runtime::RuntimeProvider + 'static>(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
    open: impl FnOnce() -> quantinfer::kernel::Result<B>,
    device: quantinfer::core::Device,
) -> Result<Box<dyn Generator>, String> {
    let manifest = manifest.or(reader.embedded_manifest());
    if let Some(manifest) = manifest {
        quantinfer_runtime::validate_manifest_for_runtime(manifest)
            .map_err(|error| error.to_string())?;
        reader
            .validate_manifest(manifest)
            .map_err(|error| error.to_string())?;
    }
    let backend = open().map_err(|error| error.to_string())?;
    gpu_plan::validate_on_device(reader, manifest, &backend, device)?;
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
