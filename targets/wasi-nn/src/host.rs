use std::path::Path;
use std::sync::Arc;

use quantinfer::cpu::{load_gguf_tokenizer, CpuModel, ModelSource};
use quantinfer::format::{ModelManifest, PackedTensorReader};
use quantinfer::runtime::{DecodeSession, GenerationConfig};
use quantinfer::tokenizer::{GgufBpeTokenizer, Tokenizer};
use wasmtime_wasi_nn::backend::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
    NamedTensor,
};
use wasmtime_wasi_nn::wit::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use wasmtime_wasi_nn::{ExecutionContext, Graph};

type Result<T> = std::result::Result<T, BackendError>;

#[derive(Debug, Clone, Copy)]
pub struct WasiNnLimits {
    pub model_bytes: usize,
    pub prompt_bytes: usize,
    pub context_tokens: usize,
    pub generation_tokens: usize,
}

impl Default for WasiNnLimits {
    fn default() -> Self {
        Self {
            model_bytes: usize::try_from(4_u64 * 1024 * 1024 * 1024).unwrap_or(usize::MAX),
            prompt_bytes: 1024 * 1024,
            context_tokens: 4096,
            generation_tokens: 128,
        }
    }
}

/// Wasmtime's GGML graph backend for QuantInfer CPU decoders.
#[derive(Debug, Default)]
pub struct QuantInferBackend {
    limits: WasiNnLimits,
}

impl QuantInferBackend {
    pub fn new(limits: WasiNnLimits) -> Result<Self> {
        if limits.model_bytes == 0
            || limits.prompt_bytes == 0
            || limits.generation_tokens == 0
            || limits.context_tokens <= limits.generation_tokens
            || limits.context_tokens > i32::MAX as usize
        {
            return Err(invalid(
                "invalid QuantInfer WASI-NN model or generation limits",
            ));
        }
        Ok(Self { limits })
    }

    /// Preload a GGUF, QIF, or verified model package on the host.
    pub fn load_path(&self, path: impl AsRef<Path>) -> Result<Graph> {
        let source = ModelSource::open(path).map_err(invalid)?;
        let reader = source
            .open_reader(Some(self.limits.model_bytes))
            .map_err(invalid)?;
        self.graph(&reader, source.package().map(|package| &package.manifest))
    }

    fn graph(
        &self,
        reader: &PackedTensorReader,
        manifest: Option<&ModelManifest>,
    ) -> Result<Graph> {
        let decoder = CpuModel::from_gguf(reader, manifest, None).map_err(invalid)?;
        let tokenizer = reader
            .metadata()
            .contains_key("tokenizer.ggml.tokens")
            .then(|| load_gguf_tokenizer(reader))
            .transpose()
            .map_err(invalid)?;
        let model = Arc::new(LoadedModel {
            decoder,
            tokenizer,
            limits: self.limits,
        });
        let graph: Box<dyn BackendGraph> = Box::new(LoadedGraph(model));
        Ok(graph.into())
    }
}

impl BackendInner for QuantInferBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Ggml
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph> {
        require_cpu(target)?;
        let [bytes] = builders else {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()));
        };
        if bytes.len() > self.limits.model_bytes {
            return Err(BackendError::NotEnoughMemory(bytes.len()));
        }
        let mut artifact = Vec::new();
        artifact.try_reserve_exact(bytes.len()).map_err(invalid)?;
        artifact.extend_from_slice(bytes);
        let reader = PackedTensorReader::open_artifact(artifact).map_err(invalid)?;
        self.graph(&reader, None)
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for QuantInferBackend {
    fn load_from_dir(&mut self, builders: &Path, target: ExecutionTarget) -> Result<Graph> {
        require_cpu(target)?;
        self.load_path(builders)
    }
}

struct LoadedGraph(Arc<LoadedModel>);

impl BackendGraph for LoadedGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext> {
        let context: Box<dyn BackendExecutionContext> = Box::new(Context {
            model: self.0.clone(),
            inputs: Inputs::default(),
            outputs: Vec::new(),
        });
        Ok(context.into())
    }
}

struct LoadedModel {
    decoder: CpuModel,
    tokenizer: Option<GgufBpeTokenizer>,
    limits: WasiNnLimits,
}

impl LoadedModel {
    fn prompt(&self, tensor: &Tensor) -> Result<Vec<usize>> {
        let count = vector_length(tensor)?;
        if tensor.data.len() > self.limits.prompt_bytes {
            return Err(BackendError::NotEnoughMemory(tensor.data.len()));
        }
        let tokens = match tensor.ty {
            TensorType::I32 if count.checked_mul(4) == Some(tensor.data.len()) => {
                if count >= self.limits.context_tokens {
                    return Err(invalid("prompt exceeds the WASI-NN context limit"));
                }
                tensor
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| usize::try_from(i32::from_le_bytes(*bytes)).map_err(invalid))
                    .collect::<Result<Vec<_>>>()?
            }
            TensorType::U8 if count == tensor.data.len() => {
                let text = std::str::from_utf8(&tensor.data).map_err(invalid)?;
                self.tokenizer
                    .as_ref()
                    .ok_or_else(|| invalid("the model requires an I32 token prompt"))?
                    .encode(text)
                    .map_err(invalid)?
            }
            _ => {
                return Err(invalid(
                    "prompt requires an I32 token vector or a UTF-8 U8 vector",
                ))
            }
        };
        if tokens.is_empty()
            || tokens.len() >= self.limits.context_tokens
            || tokens
                .iter()
                .any(|&token| token >= self.decoder.vocab_size())
        {
            return Err(invalid(
                "prompt is empty, too long, or outside the model vocabulary",
            ));
        }
        Ok(tokens)
    }

    fn generate(&self, inputs: &Inputs) -> Result<Vec<NamedTensor>> {
        let tokens = inputs
            .prompt
            .as_ref()
            .ok_or_else(|| invalid("missing prompt tensor"))?;
        let max_new_tokens = inputs.max_new_tokens.unwrap_or(1);
        let context = tokens
            .len()
            .checked_add(max_new_tokens)
            .filter(|&context| context <= self.limits.context_tokens)
            .ok_or_else(|| {
                invalid("prompt and generated tokens exceed the WASI-NN context limit")
            })?;
        let cache = self.decoder.new_cache(context).map_err(invalid)?;
        let config = GenerationConfig {
            max_new_tokens,
            eos_token_id: self.tokenizer.as_ref().and_then(Tokenizer::eos_token_id),
            ..GenerationConfig::default()
        };
        let mut session =
            DecodeSession::new(&self.decoder, cache, tokens.clone(), config).map_err(invalid)?;
        let mut tokens = Vec::new();
        let mut bytes = Vec::new();
        while let Some(token) = session.step().map_err(invalid)? {
            bytes.extend_from_slice(&i32::try_from(token).map_err(invalid)?.to_le_bytes());
            tokens.push(token);
        }
        let mut outputs = vec![NamedTensor {
            name: "tokens".to_owned(),
            tensor: Tensor::new(
                vec![u32::try_from(tokens.len()).map_err(invalid)?],
                TensorType::I32,
                bytes,
            ),
        }];
        if let Some(tokenizer) = &self.tokenizer {
            let text = tokenizer.decode(&tokens).map_err(invalid)?;
            outputs.push(NamedTensor {
                name: "text".to_owned(),
                tensor: Tensor::new(
                    vec![u32::try_from(text.len()).map_err(invalid)?],
                    TensorType::U8,
                    text.into_bytes(),
                ),
            });
        }
        Ok(outputs)
    }
}

#[derive(Default)]
struct Inputs {
    prompt: Option<Vec<usize>>,
    max_new_tokens: Option<usize>,
}

impl Inputs {
    fn set(&mut self, model: &LoadedModel, id: Id, tensor: &Tensor, unique: bool) -> Result<()> {
        if id.index() == Some(0) || id.name() == Some("prompt") {
            if unique && self.prompt.is_some() {
                return Err(invalid("duplicate prompt tensor"));
            }
            self.prompt = Some(model.prompt(tensor)?);
        } else if id.index() == Some(1) || id.name() == Some("max_new_tokens") {
            if unique && self.max_new_tokens.is_some() {
                return Err(invalid("duplicate max_new_tokens tensor"));
            }
            let bytes: [u8; 4] = tensor
                .data
                .as_slice()
                .try_into()
                .map_err(|_| invalid("max_new_tokens requires one I32 value"))?;
            if tensor.ty != TensorType::I32 || tensor.dimensions != [1] {
                return Err(invalid("max_new_tokens requires one I32 value"));
            }
            let count = usize::try_from(i32::from_le_bytes(bytes)).map_err(invalid)?;
            if count == 0 || count > model.limits.generation_tokens {
                return Err(invalid(
                    "max_new_tokens exceeds the configured generation range",
                ));
            }
            self.max_new_tokens = Some(count);
        } else {
            return Err(invalid("unknown QuantInfer input tensor"));
        }
        Ok(())
    }
}

struct Context {
    model: Arc<LoadedModel>,
    inputs: Inputs,
    outputs: Vec<NamedTensor>,
}

impl BackendExecutionContext for Context {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<()> {
        self.inputs.set(&self.model, id, tensor, false)?;
        self.outputs.clear();
        Ok(())
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor> {
        let output = match id {
            Id::Index(index) => self.outputs.get(index as usize),
            Id::Name(name) => self.outputs.iter().find(|output| output.name == name),
        };
        output
            .map(|output| output.tensor.clone())
            .ok_or_else(|| invalid("output tensor is unavailable"))
    }

    fn compute(&mut self, inputs: Option<Vec<NamedTensor>>) -> Result<Option<Vec<NamedTensor>>> {
        self.outputs.clear();
        if let Some(inputs) = inputs {
            let mut named = Inputs::default();
            for input in inputs {
                named.set(&self.model, Id::Name(input.name), &input.tensor, true)?;
            }
            self.model.generate(&named).map(Some)
        } else {
            self.outputs = self.model.generate(&self.inputs)?;
            Ok(None)
        }
    }
}

fn vector_length(tensor: &Tensor) -> Result<usize> {
    match tensor.dimensions.as_slice() {
        [length] if *length > 0 => Ok(*length as usize),
        _ => Err(invalid("tensor must be a nonempty vector")),
    }
}

fn require_cpu(target: ExecutionTarget) -> Result<()> {
    if target == ExecutionTarget::Cpu {
        Ok(())
    } else {
        Err(invalid("QuantInfer WASI-NN uses the CPU execution target"))
    }
}

fn invalid(message: impl std::fmt::Display) -> BackendError {
    BackendError::BackendAccess(wasmtime::Error::msg(message.to_string()))
}
