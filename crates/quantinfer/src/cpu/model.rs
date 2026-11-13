//! Shared CPU model and artifact loading.

use std::fs;
use std::path::{Path, PathBuf};

use quantinfer_core::Tensor;
use quantinfer_format::{GgufReader, GgufValue, ModelManifest, ModelPackage};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::ModelSpec;
use quantinfer_runtime::{
    DecodeBatchItem, DecoderModel, InferenceProfile, KvCache, LlamaModel, LoraAdapter,
    ProjectionObserver, Qwen3Model, Result, RuntimeError,
};
use quantinfer_tokenizer::GgufBpeTokenizer;

use super::{resolve_backend, resolve_profiled_backend, CpuScalar};

pub enum DenseModel<B> {
    Qwen3(Qwen3Model<B>),
    Llama(LlamaModel<B>),
}

pub type CpuModel = DenseModel<CpuScalar>;

impl DenseModel<CpuScalar> {
    pub fn spec_from_gguf(
        reader: &GgufReader,
        manifest: Option<&ModelManifest>,
    ) -> Result<ModelSpec> {
        match model_architecture(reader, manifest)? {
            "qwen3" => Qwen3Model::<CpuScalar>::spec_from_gguf(reader),
            "llama" => LlamaModel::<CpuScalar>::spec_from_gguf(reader),
            other => Err(RuntimeError::InvalidInput(format!(
                "unsupported GGUF architecture {other}"
            ))),
        }
    }

    pub fn from_gguf(
        reader: &GgufReader,
        manifest: Option<&ModelManifest>,
        profile: Option<&InferenceProfile>,
    ) -> Result<Self> {
        let architecture = model_architecture(reader, manifest)?;
        let manifest = manifest.or(reader.embedded_manifest());
        let backend = if let Some(profile) = profile {
            let manifest = manifest.ok_or_else(|| {
                RuntimeError::InvalidInput("inference profiles require a model manifest".to_owned())
            })?;
            resolve_profiled_backend(reader, manifest, profile)
        } else {
            resolve_backend(reader, manifest)
        }
        .map_err(|error| {
            RuntimeError::InvalidInput(format!("cannot resolve CPU execution provider: {error}"))
        })?;
        Self::load(backend, reader, architecture)
    }
}

impl<B: KernelBackend> DenseModel<B> {
    pub fn with_backend(
        backend: B,
        reader: &GgufReader,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        let architecture = model_architecture(reader, manifest)?;
        Self::load(backend, reader, architecture)
    }

    fn load(backend: B, reader: &GgufReader, architecture: &str) -> Result<Self> {
        match architecture {
            "qwen3" => Qwen3Model::from_gguf(backend, reader)
                .map(Self::Qwen3)
                .map_err(|error| {
                    RuntimeError::InvalidInput(format!("cannot load Qwen3 model: {error}"))
                }),
            "llama" => LlamaModel::from_gguf(backend, reader)
                .map(Self::Llama)
                .map_err(|error| {
                    RuntimeError::InvalidInput(format!("cannot load Llama model: {error}"))
                }),
            other => Err(RuntimeError::InvalidInput(format!(
                "unsupported GGUF architecture {other}"
            ))),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen3(model) => model.config.vocab_size,
            Self::Llama(model) => model.config.vocab_size,
        }
    }

    pub fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Qwen3(model) => model.config.max_position_embeddings,
            Self::Llama(model) => model.config.max_position_embeddings,
        }
    }

    pub fn observe_projections(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
        observer: &mut dyn ProjectionObserver,
    ) -> Result<()> {
        match self {
            Self::Qwen3(model) => model.observe_projections(token_ids, cache, observer),
            Self::Llama(model) => model.observe_projections(token_ids, cache, observer),
        }
    }

    pub fn attach_lora(&mut self, adapter: LoraAdapter) -> Result<()> {
        match self {
            Self::Qwen3(model) => model.attach_lora(adapter),
            Self::Llama(model) => model.attach_lora(adapter),
        }
    }

    pub fn new_cache(&self, context_length: usize) -> Result<KvCache> {
        if context_length > self.max_position_embeddings() {
            return Err(RuntimeError::InvalidInput(
                "requested context exceeds model capacity".to_owned(),
            ));
        }
        let (layers, heads, head_dim) = match self {
            Self::Qwen3(model) => (
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.effective_head_dim(),
            ),
            Self::Llama(model) => (
                model.config.num_hidden_layers,
                model.config.num_key_value_heads(),
                model.config.effective_head_dim(),
            ),
        };
        KvCache::new(layers, context_length, heads, head_dim)
    }
}

impl<B: KernelBackend> DecoderModel for DenseModel<B> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        match self {
            Self::Qwen3(model) => model.forward(token_ids, cache),
            Self::Llama(model) => model.forward(token_ids, cache),
        }
    }

    fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        match self {
            Self::Qwen3(model) => model.forward_batch(batch),
            Self::Llama(model) => model.forward_batch(batch),
        }
    }
}

pub fn model_architecture<'a>(
    reader: &'a GgufReader,
    manifest: Option<&'a ModelManifest>,
) -> Result<&'a str> {
    let archive_architecture = match reader.metadata().get("general.architecture") {
        Some(GgufValue::String(value)) => Some(value.as_str()),
        Some(_) => {
            return Err(RuntimeError::InvalidInput(
                "GGUF general.architecture must be a string".to_owned(),
            ))
        }
        None => reader
            .embedded_manifest()
            .map(|manifest| manifest.architecture.as_str()),
    };
    match manifest {
        Some(manifest) => {
            if archive_architecture
                .is_some_and(|architecture| architecture != manifest.architecture)
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "model package architecture {} disagrees with the tensor file",
                    manifest.architecture
                )));
            }
            Ok(&manifest.architecture)
        }
        None => Ok(archive_architecture.unwrap_or("qwen3")),
    }
}

pub enum ModelSource {
    Package(Box<ModelPackage>),
    Gguf(PathBuf),
}

impl ModelSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            ModelPackage::open(path)
                .map(|package| Self::Package(Box::new(package)))
                .map_err(|error| {
                    RuntimeError::InvalidInput(format!(
                        "cannot open model package {}: {error}",
                        path.display()
                    ))
                })
        } else {
            Ok(Self::Gguf(path.to_path_buf()))
        }
    }

    pub fn package(&self) -> Option<&ModelPackage> {
        match self {
            Self::Package(package) => Some(package),
            Self::Gguf(_) => None,
        }
    }

    pub fn open_reader(&self, budget: Option<usize>) -> Result<GgufReader> {
        match self {
            Self::Package(package) => {
                let model_name = package
                    .manifest
                    .checksums
                    .keys()
                    .find(|name| name.ends_with(".gguf"))
                    .or_else(|| {
                        package
                            .manifest
                            .checksums
                            .keys()
                            .find(|name| name.ends_with(".qif"))
                    })
                    .map(String::as_str)
                    .or_else(|| {
                        package
                            .root()
                            .join("model.gguf")
                            .exists()
                            .then_some("model.gguf")
                    })
                    .ok_or_else(|| {
                        RuntimeError::InvalidInput(
                            "model package manifest does not identify a GGUF or QIF file"
                                .to_owned(),
                        )
                    })?;
                let model_path = package
                    .artifact_path(model_name)
                    .map_err(|error| RuntimeError::InvalidInput(error.to_string()))?;
                check_file_budget(&model_path, budget)?;
                package.open_verified_artifact(model_name).map_err(|error| {
                    RuntimeError::InvalidInput(format!("cannot parse packaged model: {error}"))
                })
            }
            Self::Gguf(path) => {
                check_file_budget(path, budget)?;
                GgufReader::open_artifact_path(path).map_err(|error| {
                    RuntimeError::InvalidInput(format!(
                        "cannot parse model {}: {error}",
                        path.display()
                    ))
                })
            }
        }
    }
}

fn check_file_budget(path: &Path, budget: Option<usize>) -> Result<()> {
    if let Some(budget) = budget {
        let file_size = fs::metadata(path)
            .map_err(|error| {
                RuntimeError::InvalidInput(format!("cannot stat {}: {error}", path.display()))
            })?
            .len();
        if file_size > budget as u64 {
            return Err(RuntimeError::InvalidInput(format!(
                "memory budget check failed: model file is {file_size} bytes, budget is {budget}"
            )));
        }
    }
    Ok(())
}

pub fn load_gguf_tokenizer(reader: &GgufReader) -> Result<GgufBpeTokenizer> {
    let tokens = metadata_string_array(reader, "tokenizer.ggml.tokens")?;
    let merges = metadata_string_array_or_empty(reader, "tokenizer.ggml.merges")?;
    let pre_tokenizer = metadata_string_optional(reader, "tokenizer.ggml.pre")?;
    let bos_token_id = metadata_usize(reader, "tokenizer.ggml.bos_token_id")?;
    let eos_token_id = metadata_usize(reader, "tokenizer.ggml.eos_token_id")?;
    GgufBpeTokenizer::new_with_pre(
        tokens,
        &merges,
        bos_token_id,
        eos_token_id,
        pre_tokenizer.as_deref(),
    )
    .map_err(|error| RuntimeError::InvalidInput(format!("cannot build GGUF tokenizer: {error}")))
}

fn metadata_string_array(reader: &GgufReader, key: &str) -> Result<Vec<String>> {
    let value = reader.metadata().get(key).ok_or_else(|| {
        RuntimeError::InvalidInput(format!("GGUF is missing tokenizer metadata {key}"))
    })?;
    let invalid =
        || RuntimeError::InvalidInput(format!("GGUF metadata {key} must be an array of strings"));
    match value {
        GgufValue::Array(values) => values
            .iter()
            .map(|value| match value {
                GgufValue::String(value) => Ok(value.clone()),
                _ => Err(invalid()),
            })
            .collect(),
        _ => Err(invalid()),
    }
}

fn metadata_string_array_or_empty(reader: &GgufReader, key: &str) -> Result<Vec<String>> {
    if reader.metadata().contains_key(key) {
        metadata_string_array(reader, key)
    } else {
        Ok(Vec::new())
    }
}

fn metadata_string_optional(reader: &GgufReader, key: &str) -> Result<Option<String>> {
    let Some(value) = reader.metadata().get(key) else {
        return Ok(None);
    };
    match value {
        GgufValue::String(value) => Ok(Some(value.clone())),
        _ => Err(RuntimeError::InvalidInput(format!(
            "GGUF metadata {key} must be a string"
        ))),
    }
}

fn metadata_usize(reader: &GgufReader, key: &str) -> Result<Option<usize>> {
    let Some(value) = reader.metadata().get(key) else {
        return Ok(None);
    };
    let overflow = |_| RuntimeError::InvalidInput(format!("GGUF metadata {key} overflows usize"));
    match value {
        GgufValue::U32(value) => Ok(Some(*value as usize)),
        GgufValue::U64(value) => usize::try_from(*value).map(Some).map_err(overflow),
        GgufValue::I32(value) if *value >= 0 => Ok(Some(*value as usize)),
        GgufValue::I64(value) if *value >= 0 => usize::try_from(*value).map(Some).map_err(overflow),
        _ => Err(RuntimeError::InvalidInput(format!(
            "GGUF metadata {key} must be an unsigned integer"
        ))),
    }
}
