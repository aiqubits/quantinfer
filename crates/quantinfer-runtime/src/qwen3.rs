use quantinfer_core::{
    DType, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K,
    QuantizedTensorQ6K, Shape, Tensor,
};
use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::{ModelSpec, Qwen3Adapter, Qwen3Config, TensorRole};

use crate::batch::{last_hidden_rows, DenseAttention, DenseBatchLimits};
use crate::calibration::{ProjectionObserver, ProjectionTrace};
use crate::llama::{load_ggml_as_float, load_ggml_linear, load_ggml_matrix};
use crate::{DecodeBatchItem, KvCache, LoraAdapter, LoraProjection, Result, RuntimeError};

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum LinearWeight {
    F32(Tensor),
    Q8_0(QuantizedTensor),
    Q4_0(QuantizedTensor4),
    Q4_K(QuantizedTensorQ4K),
    Q5_K(QuantizedTensorQ5K),
    Q6_K(QuantizedTensorQ6K),
    Q8_0OutputRows(QuantizedTensor),
    Q4_0OutputRows(QuantizedTensor4),
    Lora {
        base: Box<LinearWeight>,
        projection: LoraProjection,
    },
}

impl LinearWeight {
    pub fn shape(&self) -> &Shape {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Q8_0(tensor) | Self::Q8_0OutputRows(tensor) => tensor.shape(),
            Self::Q4_0(tensor) | Self::Q4_0OutputRows(tensor) => tensor.shape(),
            Self::Q4_K(tensor) => tensor.shape(),
            Self::Q5_K(tensor) => tensor.shape(),
            Self::Q6_K(tensor) => tensor.shape(),
            Self::Lora { base, .. } => base.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::Q8_0(_) | Self::Q8_0OutputRows(_) => DType::Q8_0,
            Self::Q4_0(_) | Self::Q4_0OutputRows(_) => DType::Q4_0,
            Self::Q4_K(_) => DType::Q4_K,
            Self::Q5_K(_) => DType::Q5_K,
            Self::Q6_K(_) => DType::Q6_K,
            Self::Lora { base, .. } => base.dtype(),
        }
    }

    pub(crate) fn matmul<B: KernelBackend>(&self, backend: &B, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::F32(weight) => Ok(backend.matmul(input, weight)?),
            Self::Q8_0(weight) => Ok(backend.matmul_q8_0(input, weight)?),
            Self::Q4_0(weight) => Ok(backend.matmul_q4_0(input, weight)?),
            Self::Q4_K(weight) => Ok(backend.matmul_q4_k(input, weight)?),
            Self::Q5_K(weight) => Ok(backend.matmul_q5_k(input, weight)?),
            Self::Q6_K(weight) => Ok(backend.matmul_q6_k(input, weight)?),
            Self::Q8_0OutputRows(weight) => Ok(backend.matmul_q8_0_output_rows(input, weight)?),
            Self::Q4_0OutputRows(weight) => Ok(backend.matmul_q4_0_output_rows(input, weight)?),
            Self::Lora { base, projection } => {
                projection.add_to(backend, input, base.matmul(backend, input)?)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3LayerWeights {
    pub input_layernorm: Tensor,
    pub q_proj: LinearWeight,
    pub q_norm: Tensor,
    pub k_proj: LinearWeight,
    pub k_norm: Tensor,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub post_attention_layernorm: Tensor,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

impl Qwen3LayerWeights {
    pub fn validate(&self, config: &Qwen3Config) -> Result<()> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query = config.query_projection_size();
        let key_value = config.key_value_projection_size();
        let expected = [
            (
                "input_layernorm",
                self.input_layernorm.shape().dims(),
                &[hidden][..],
            ),
            ("q_proj", self.q_proj.shape().dims(), &[hidden, query][..]),
            (
                "q_norm",
                self.q_norm.shape().dims(),
                &[config.effective_head_dim()][..],
            ),
            (
                "k_proj",
                self.k_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            (
                "k_norm",
                self.k_norm.shape().dims(),
                &[config.effective_head_dim()][..],
            ),
            (
                "v_proj",
                self.v_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            ("o_proj", self.o_proj.shape().dims(), &[query, hidden][..]),
            (
                "post_attention_layernorm",
                self.post_attention_layernorm.shape().dims(),
                &[hidden][..],
            ),
            (
                "gate_proj",
                self.gate_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "up_proj",
                self.up_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "down_proj",
                self.down_proj.shape().dims(),
                &[intermediate, hidden][..],
            ),
        ];
        for (name, actual, expected) in expected {
            if actual != expected {
                return Err(RuntimeError::InvalidInput(format!(
                    "Qwen3 {name} has shape {actual:?}, expected {expected:?}"
                )));
            }
            if !matches!(
                name,
                "q_proj" | "k_proj" | "v_proj" | "o_proj" | "gate_proj" | "up_proj" | "down_proj"
            ) && self.weight_dtype(name)? != DType::F32
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "Qwen3 {name} must use F32 for the reference runtime"
                )));
            }
        }
        Ok(())
    }

    fn weight_dtype(&self, name: &str) -> Result<DType> {
        match name {
            "input_layernorm" => Ok(self.input_layernorm.dtype()),
            "q_proj" => Ok(self.q_proj.dtype()),
            "q_norm" => Ok(self.q_norm.dtype()),
            "k_proj" => Ok(self.k_proj.dtype()),
            "k_norm" => Ok(self.k_norm.dtype()),
            "v_proj" => Ok(self.v_proj.dtype()),
            "o_proj" => Ok(self.o_proj.dtype()),
            "post_attention_layernorm" => Ok(self.post_attention_layernorm.dtype()),
            "gate_proj" => Ok(self.gate_proj.dtype()),
            "up_proj" => Ok(self.up_proj.dtype()),
            "down_proj" => Ok(self.down_proj.dtype()),
            _ => Err(RuntimeError::InvalidInput(format!(
                "unknown Qwen3 weight {name}"
            ))),
        }
    }

    fn forward<B: KernelBackend>(
        &self,
        backend: &B,
        config: &Qwen3Config,
        hidden: &Tensor,
        mut trace: ProjectionTrace<'_, '_>,
        attention: &DenseAttention,
        batch: &mut [DecodeBatchItem<'_>],
    ) -> Result<Tensor> {
        let sequence_length = hidden.shape().dims()[0];
        let residual = hidden.clone();
        let normalized = backend.rms_norm(hidden, &self.input_layernorm, config.rms_norm_eps)?;
        trace.record(
            &[
                TensorRole::QueryProjection,
                TensorRole::KeyProjection,
                TensorRole::ValueProjection,
            ],
            &normalized,
        )?;
        let query = self
            .q_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_attention_heads,
                config.effective_head_dim(),
            ]))?;
        let key = self
            .k_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads,
                config.effective_head_dim(),
            ]))?;
        let value = self
            .v_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads,
                config.effective_head_dim(),
            ]))?;
        let query = backend.rms_norm_heads(&query, &self.q_norm, config.rms_norm_eps)?;
        let key = backend.rms_norm_heads(&key, &self.k_norm, config.rms_norm_eps)?;
        let attended = attention
            .apply(backend, &query, &key, &value, trace.layer, batch)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.query_projection_size(),
            ]))?;
        trace.record(&[TensorRole::OutputProjection], &attended)?;
        let projected = self.o_proj.matmul(backend, &attended)?;
        let residual = add(&residual, &projected)?;
        let normalized = backend.rms_norm(
            &residual,
            &self.post_attention_layernorm,
            config.rms_norm_eps,
        )?;
        trace.record(
            &[TensorRole::GateProjection, TensorRole::UpProjection],
            &normalized,
        )?;
        let gate = self.gate_proj.matmul(backend, &normalized)?;
        let up = self.up_proj.matmul(backend, &normalized)?;
        let feed_forward = backend.silu_mul(&gate, &up)?;
        trace.record(&[TensorRole::DownProjection], &feed_forward)?;
        let feed_forward = self.down_proj.matmul(backend, &feed_forward)?;
        add(&residual, &feed_forward)
    }
}

pub struct Qwen3Model<B> {
    pub config: Qwen3Config,
    backend: B,
    pub embedding: Tensor,
    pub layers: Vec<Qwen3LayerWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
    lora_head: Option<LoraProjection>,
}

impl<B: KernelBackend> Qwen3Model<B> {
    pub fn spec_from_gguf(reader: &GgufReader) -> Result<ModelSpec> {
        config_from_gguf(reader, reader.uses_ggml_output_rows("qwen3")?)?
            .to_model_spec("qwen3")
            .map_err(RuntimeError::InvalidInput)
    }

    pub fn new(
        config: Qwen3Config,
        backend: B,
        embedding: Tensor,
        layers: Vec<Qwen3LayerWeights>,
        final_norm: Tensor,
        lm_head: Tensor,
    ) -> Result<Self> {
        config.validate().map_err(RuntimeError::InvalidInput)?;
        expect_shape(
            &embedding,
            &[config.vocab_size, config.hidden_size],
            "embedding",
        )?;
        expect_shape(&final_norm, &[config.hidden_size], "final_norm")?;
        expect_shape(
            &lm_head,
            &[config.hidden_size, config.vocab_size],
            "lm_head",
        )?;
        if layers.len() != config.num_hidden_layers {
            return Err(RuntimeError::InvalidInput(format!(
                "Qwen3 has {} layers but received {} weight sets",
                config.num_hidden_layers,
                layers.len()
            )));
        }
        for layer in &layers {
            layer.validate(&config)?;
        }
        Ok(Self {
            config,
            backend,
            embedding,
            layers,
            final_norm,
            lm_head,
            lora_head: None,
        })
    }

    /// Attach low-rank weights before populating a KV cache for the adapted model.
    pub fn attach_lora(&mut self, adapter: LoraAdapter) -> Result<()> {
        let mut projections = self
            .layers
            .iter_mut()
            .map(|layer| {
                [
                    &mut layer.q_proj,
                    &mut layer.k_proj,
                    &mut layer.v_proj,
                    &mut layer.o_proj,
                    &mut layer.gate_proj,
                    &mut layer.up_proj,
                    &mut layer.down_proj,
                ]
            })
            .collect::<Vec<_>>();
        adapter.install(&mut projections, &self.lm_head, &mut self.lora_head)
    }

    pub fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.forward_sequence(token_ids, cache, None)?;
        self.project_logits(&hidden)
    }

    /// Observe layer projection inputs while advancing the supplied KV cache.
    pub fn observe_projections(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
        observer: &mut dyn ProjectionObserver,
    ) -> Result<()> {
        self.forward_sequence(token_ids, cache, Some(observer))
            .map(drop)
    }

    fn forward_sequence(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
        observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(RuntimeError::InvalidInput(
                "Qwen3 forward requires at least one token".to_owned(),
            ));
        }
        if cache.sequence_length(0)? + token_ids.len() > self.config.max_position_embeddings {
            return Err(RuntimeError::InvalidInput(
                "input exceeds Qwen3 max_position_embeddings".to_owned(),
            ));
        }
        let mut request = [DecodeBatchItem {
            tokens: token_ids,
            cache,
        }];
        self.forward_hidden(token_ids, &mut request, observer)
    }

    pub fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        let limits = DenseBatchLimits {
            vocab_size: self.config.vocab_size,
            max_positions: self.config.max_position_embeddings,
            layers: self.layers.len(),
            key_value_heads: self.config.num_key_value_heads,
            head_dim: self.config.effective_head_dim(),
        };
        limits.run(batch, |tokens, requests| {
            let hidden = self.forward_hidden(tokens, requests, None)?;
            self.project_logits(&last_hidden_rows(&hidden, requests)?)
        })
    }

    fn forward_hidden(
        &self,
        token_ids: &[usize],
        batch: &mut [DecodeBatchItem<'_>],
        mut observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        let attention = DenseAttention {
            query_heads: self.config.num_attention_heads,
            key_value_heads: self.config.num_key_value_heads,
            head_dim: self.config.effective_head_dim(),
            rope_theta: self.config.rope_theta,
        };
        let mut hidden = self.backend.embedding(&self.embedding, token_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &self.backend,
                &self.config,
                &hidden,
                ProjectionTrace {
                    layer: index,
                    observer: &mut observer,
                },
                &attention,
                batch,
            )?;
        }
        Ok(hidden)
    }

    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self
            .backend
            .rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let logits = self.backend.matmul(&hidden, &self.lm_head)?;
        match &self.lora_head {
            Some(adapter) => adapter.add_to(&self.backend, &hidden, logits),
            None => Ok(logits),
        }
    }

    pub fn from_gguf(backend: B, reader: &GgufReader) -> Result<Self> {
        let ggml_layout = reader.uses_ggml_output_rows("qwen3")?;
        let config = config_from_gguf(reader, ggml_layout)?;
        let adapter =
            Qwen3Adapter::from_config(config.clone()).map_err(RuntimeError::InvalidInput)?;
        let spec = adapter.spec();
        let embedding_name = crate::tensor_source(spec, TensorRole::TokenEmbedding, None)?;
        let embedding = if ggml_layout {
            let embedding = load_ggml_as_float(reader, embedding_name)?;
            expect_shape(
                &embedding,
                &[config.hidden_size, config.vocab_size],
                "GGML embedding",
            )?;
            embedding.reshape(spec.embedding.shape.clone())?
        } else {
            reader.tensor_f32(embedding_name)?
        };
        let final_norm = reader.tensor_f32(crate::tensor_source(
            spec,
            TensorRole::FinalNormalization,
            None,
        )?)?;
        let head_name = crate::lm_head_source(&adapter)?;
        let lm_head = match reader.tensor_info(&head_name) {
            Ok(_) if ggml_layout => load_ggml_matrix(reader, &head_name, None)?,
            Ok(info) if matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) => {
                reader.tensor_f32(&head_name)?
            }
            Ok(info) => {
                return Err(RuntimeError::InvalidInput(format!(
                    "GGUF LM head {head_name} has unsupported type {:?}",
                    info.dtype
                )))
            }
            Err(_) if spec.embedding.tied_lm_head => transpose_matrix(&embedding)?,
            Err(error) => return Err(error.into()),
        };
        let mut layers = Vec::with_capacity(spec.layers.len());
        for layer in &spec.layers {
            let name = |role| crate::tensor_source(spec, role, Some(layer.index));
            let float = |role| Ok::<_, RuntimeError>(reader.tensor_f32(name(role)?)?);
            let linear = |role| {
                let name = name(role)?;
                if ggml_layout {
                    load_ggml_linear(reader, name, None)
                } else {
                    load_linear(reader, name)
                }
            };
            layers.push(Qwen3LayerWeights {
                input_layernorm: float(TensorRole::InputNormalization)?,
                q_proj: linear(TensorRole::QueryProjection)?,
                q_norm: float(TensorRole::QueryNormalization)?,
                k_proj: linear(TensorRole::KeyProjection)?,
                k_norm: float(TensorRole::KeyNormalization)?,
                v_proj: linear(TensorRole::ValueProjection)?,
                o_proj: linear(TensorRole::OutputProjection)?,
                post_attention_layernorm: float(TensorRole::PostAttentionNormalization)?,
                gate_proj: linear(TensorRole::GateProjection)?,
                up_proj: linear(TensorRole::UpProjection)?,
                down_proj: linear(TensorRole::DownProjection)?,
            });
        }
        Self::new(config, backend, embedding, layers, final_norm, lm_head)
    }
}

pub(crate) fn load_linear(reader: &GgufReader, name: &str) -> Result<LinearWeight> {
    match reader.tensor_info(name)?.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            Ok(LinearWeight::F32(reader.tensor_f32(name)?))
        }
        GgmlType::Q8_0 => Ok(LinearWeight::Q8_0(reader.tensor_q8_0(name)?)),
        GgmlType::Q4_0 => Ok(LinearWeight::Q4_0(reader.tensor_q4_0(name)?)),
        GgmlType::Q4_K => Ok(LinearWeight::Q4_K(reader.tensor_q4_k(name)?)),
        GgmlType::Q6_K => Ok(LinearWeight::Q6_K(reader.tensor_q6_k(name)?)),
        GgmlType::Q5_K => Ok(LinearWeight::Q5_K(reader.tensor_q5_k(name)?)),
    }
}

pub(crate) fn transpose_matrix(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(RuntimeError::InvalidInput(
            "tied Qwen3 embedding must be a matrix".to_owned(),
        ));
    }
    let rows = dims[0];
    let columns = dims[1];
    let source = tensor.as_f32()?;
    let mut values = vec![0.0; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            values[column * rows + row] = source[row * columns + column];
        }
    }
    Ok(Tensor::from_f32(vec![columns, rows], values)?)
}

fn config_from_gguf(reader: &GgufReader, ggml_layout: bool) -> Result<Qwen3Config> {
    let embedding = reader.tensor_info("token_embd.weight")?;
    let (hidden_axis, vocab_axis) = if ggml_layout { (0, 1) } else { (1, 0) };
    let vocab_size = metadata_usize(reader, "qwen3.vocab_size")?
        .unwrap_or_else(|| embedding.shape.get(vocab_axis).copied().unwrap_or_default());
    let hidden_size = metadata_usize(reader, "qwen3.embedding_length")?.unwrap_or_else(|| {
        embedding
            .shape
            .get(hidden_axis)
            .copied()
            .unwrap_or_default()
    });
    let num_hidden_layers = required_metadata_usize(reader, "qwen3.block_count")?;
    let num_attention_heads = required_metadata_usize(reader, "qwen3.attention.head_count")?;
    let num_key_value_heads =
        metadata_usize(reader, "qwen3.attention.head_count_kv")?.unwrap_or(num_attention_heads);
    let intermediate_size = metadata_usize(reader, "qwen3.feed_forward_length")?.or_else(|| {
        reader
            .tensor_info("blk.0.ffn_gate.weight")
            .ok()
            .and_then(|info| info.shape.get(1).copied())
    });
    let intermediate_size = intermediate_size.ok_or_else(|| {
        RuntimeError::InvalidInput(
            "GGUF is missing qwen3.feed_forward_length and ffn_gate shape".to_owned(),
        )
    })?;
    let max_position_embeddings = required_metadata_usize(reader, "qwen3.context_length")?;
    let head_dim = if ggml_layout {
        metadata_usize(reader, "qwen3.attention.key_length")?
    } else {
        metadata_usize(reader, "qwen3.attention.head_dim")?
    };
    let rms_norm_eps =
        metadata_f32(reader, "qwen3.attention.layer_norm_rms_epsilon")?.unwrap_or(1e-6);
    let rope_theta = metadata_f32(reader, "qwen3.rope.freq_base")?.unwrap_or(1_000_000.0);
    let tie_word_embeddings = metadata_bool(reader, "qwen3.tie_word_embeddings")?
        .unwrap_or_else(|| ggml_layout && reader.tensor_info("output.weight").is_err());
    let architectures = metadata_architectures(reader)
        .or_else(|| metadata_architectures_from_key(reader, "qwen3.architectures"))
        .unwrap_or_else(|| vec!["Qwen3ForCausalLM".to_owned()]);
    let config = Qwen3Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        rms_norm_eps,
        rope_theta,
        max_position_embeddings,
        tie_word_embeddings,
        sliding_window: None,
        max_window_layers: None,
        use_sliding_window: false,
        rope_scaling: None,
        architectures,
    };
    config.validate().map_err(RuntimeError::InvalidInput)?;
    if ggml_layout {
        for key in ["qwen3.rope.dimension_count", "qwen3.attention.value_length"] {
            if metadata_usize(reader, key)?
                .is_some_and(|value| value != config.effective_head_dim())
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "unsupported GGML metadata {key}"
                )));
            }
        }
        if reader
            .metadata()
            .get("qwen3.rope.scaling.type")
            .is_some_and(|value| !matches!(value, GgufValue::String(kind) if kind == "none"))
        {
            return Err(RuntimeError::InvalidInput(
                "custom GGML RoPE scaling is unsupported".to_owned(),
            ));
        }
    }
    Ok(config)
}

fn required_metadata_usize(reader: &GgufReader, key: &str) -> Result<usize> {
    metadata_usize(reader, key)?.ok_or_else(|| {
        RuntimeError::InvalidInput(format!("GGUF is missing required metadata {key}"))
    })
}

fn metadata_usize(reader: &GgufReader, key: &str) -> Result<Option<usize>> {
    let Some(value) = reader.metadata().get(key) else {
        return Ok(None);
    };
    let value = match value {
        GgufValue::U32(value) => *value as usize,
        GgufValue::U64(value) => usize::try_from(*value).map_err(|_| {
            RuntimeError::InvalidInput(format!("GGUF metadata {key} overflows usize"))
        })?,
        GgufValue::I32(value) if *value >= 0 => *value as usize,
        GgufValue::I64(value) if *value >= 0 => usize::try_from(*value).map_err(|_| {
            RuntimeError::InvalidInput(format!("GGUF metadata {key} overflows usize"))
        })?,
        _ => {
            return Err(RuntimeError::InvalidInput(format!(
                "GGUF metadata {key} must be an unsigned integer"
            )))
        }
    };
    Ok(Some(value))
}

fn metadata_f32(reader: &GgufReader, key: &str) -> Result<Option<f32>> {
    let Some(value) = reader.metadata().get(key) else {
        return Ok(None);
    };
    match value {
        GgufValue::F32(value) => Ok(Some(*value)),
        GgufValue::F64(value) => Ok(Some(*value as f32)),
        GgufValue::U32(value) => Ok(Some(*value as f32)),
        GgufValue::U64(value) => Ok(Some(*value as f32)),
        _ => Err(RuntimeError::InvalidInput(format!(
            "GGUF metadata {key} must be a number"
        ))),
    }
}

fn metadata_bool(reader: &GgufReader, key: &str) -> Result<Option<bool>> {
    let Some(value) = reader.metadata().get(key) else {
        return Ok(None);
    };
    match value {
        GgufValue::Bool(value) => Ok(Some(*value)),
        _ => Err(RuntimeError::InvalidInput(format!(
            "GGUF metadata {key} must be a boolean"
        ))),
    }
}

fn metadata_architectures(reader: &GgufReader) -> Option<Vec<String>> {
    metadata_architectures_from_key(reader, "general.architectures")
}

fn metadata_architectures_from_key(reader: &GgufReader, key: &str) -> Option<Vec<String>> {
    match reader.metadata().get(key) {
        Some(GgufValue::Array(values)) => Some(
            values
                .iter()
                .filter_map(|value| match value {
                    GgufValue::String(value) => Some(value.clone()),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(RuntimeError::InvalidInput(format!(
            "Qwen3 {name} has shape {:?}, expected {expected:?}",
            tensor.shape().dims()
        )));
    }
    if tensor.dtype() != quantinfer_core::DType::F32 {
        return Err(RuntimeError::InvalidInput(format!(
            "Qwen3 {name} must use F32 for the reference runtime"
        )));
    }
    Ok(())
}

pub(crate) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape() != rhs.shape() {
        return Err(RuntimeError::InvalidInput(
            "residual tensors must have identical shapes".to_owned(),
        ));
    }
    let values = lhs
        .as_f32()?
        .iter()
        .zip(rhs.as_f32()?)
        .map(|(left, right)| left + right)
        .collect();
    Ok(Tensor::from_f32(lhs.shape().clone(), values)?)
}

pub(crate) fn rope_tables(
    theta: f32,
    head_dim: usize,
    start: usize,
    length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pair_count = head_dim / 2;
    let mut cos = Vec::with_capacity(length * pair_count);
    let mut sin = Vec::with_capacity(length * pair_count);
    for position in start..start + length {
        for pair in 0..pair_count {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let angle = position as f32 * theta.powf(-exponent);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attention(
    query: &Tensor,
    cache: &mut KvCache,
    layer_index: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    start_position: usize,
    reuse_workspace: bool,
) -> Result<Tensor> {
    let layer = cache
        .layers
        .get(layer_index)
        .ok_or_else(|| RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned()))?;
    let key_values = &layer.keys;
    let value_values = &layer.values;
    let key_tokens = layer.sequence_length;
    let query_dims = query.shape().dims();
    if query_dims.len() != 3
        || query_dims[1] != query_heads
        || query_dims[2] != head_dim
        || key_tokens == 0
        || key_value_heads == 0
        || key_values.len() != key_tokens * key_value_heads * head_dim
        || value_values.len() != key_values.len()
        || !query_heads.is_multiple_of(key_value_heads)
    {
        return Err(RuntimeError::InvalidInput(
            "invalid Qwen3 GQA tensor shapes".to_owned(),
        ));
    }
    let query_values = query.as_f32()?;
    let query_tokens = query_dims[0];
    if key_tokens < start_position + query_tokens {
        return Err(RuntimeError::InvalidInput(
            "KV cache does not contain all causal attention keys".to_owned(),
        ));
    }
    let group_size = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; query_tokens * query_heads * head_dim];
    let mut local_scores = Vec::new();
    let scores = if reuse_workspace {
        &mut cache.attention_scores
    } else {
        &mut local_scores
    };
    scores.clear();
    if scores.capacity() < key_tokens {
        let target = key_tokens
            .max(scores.capacity().saturating_mul(2))
            .min(cache.max_sequence_length);
        scores.reserve_exact(target);
    }
    for token in 0..query_tokens {
        let last_key = start_position + token;
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = (token * query_heads + query_head) * head_dim;
            scores.clear();
            for key_token in 0..=last_key {
                let key_start = (key_token * key_value_heads + kv_head) * head_dim;
                let score = query_values[query_start..query_start + head_dim]
                    .iter()
                    .zip(&key_values[key_start..key_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
            let output_start = query_start;
            for (key_token, score) in scores.iter().copied().enumerate() {
                let weight = (score - maximum).exp() / denominator;
                let value_start = (key_token * key_value_heads + kv_head) * head_dim;
                for dimension in 0..head_dim {
                    output[output_start + dimension] +=
                        weight * value_values[value_start + dimension];
                }
            }
        }
    }
    Ok(Tensor::from_f32(
        Shape::from(vec![query_tokens, query_heads, head_dim]),
        output,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{LinearWeight, Qwen3LayerWeights, Qwen3Model};
    use crate::{generate_qwen3_stream, GenerationConfig, KvCache, StreamControl};
    use quantinfer_backend_cpu::{CpuRuntimeConfig, CpuScalar};
    use quantinfer_core::{DType, Tensor};
    use quantinfer_format::{GgufReader, GgufValue, GgufWriter};
    use quantinfer_model::Qwen3Config;
    use quantinfer_quant::{
        quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0,
    };

    fn filled(shape: Vec<usize>, value: f32) -> Tensor {
        let count = shape.iter().product();
        Tensor::from_f32(shape, vec![value; count]).unwrap()
    }

    fn linear(shape: Vec<usize>, value: f32) -> LinearWeight {
        LinearWeight::F32(filled(shape, value))
    }

    fn quantized_linear(shape: Vec<usize>, value: f32) -> LinearWeight {
        LinearWeight::Q8_0(quantize_q8_0(&filled(shape, value)).unwrap())
    }

    fn config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: false,
            rope_scaling: None,
            architectures: vec!["Qwen3ForCausalLM".to_owned()],
        }
    }

    fn layer() -> Qwen3LayerWeights {
        Qwen3LayerWeights {
            input_layernorm: filled(vec![8], 1.0),
            q_proj: linear(vec![8, 8], 0.01),
            q_norm: filled(vec![4], 1.0),
            k_proj: linear(vec![8, 4], 0.01),
            k_norm: filled(vec![4], 1.0),
            v_proj: linear(vec![8, 4], 0.01),
            o_proj: linear(vec![8, 8], 0.01),
            post_attention_layernorm: filled(vec![8], 1.0),
            gate_proj: linear(vec![8, 16], 0.01),
            up_proj: linear(vec![8, 16], 0.01),
            down_proj: linear(vec![16, 8], 0.01),
        }
    }

    fn quantized_layer() -> Qwen3LayerWeights {
        Qwen3LayerWeights {
            input_layernorm: filled(vec![8], 1.0),
            q_proj: quantized_linear(vec![8, 8], 0.01),
            q_norm: filled(vec![4], 1.0),
            k_proj: quantized_linear(vec![8, 4], 0.01),
            k_norm: filled(vec![4], 1.0),
            v_proj: quantized_linear(vec![8, 4], 0.01),
            o_proj: quantized_linear(vec![8, 8], 0.01),
            post_attention_layernorm: filled(vec![8], 1.0),
            gate_proj: quantized_linear(vec![8, 16], 0.01),
            up_proj: quantized_linear(vec![8, 16], 0.01),
            down_proj: quantized_linear(vec![16, 8], 0.01),
        }
    }

    #[test]
    fn runs_qwen3_prefill_and_decode() {
        let config = config();
        let model = Qwen3Model::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.effective_head_dim(),
        )
        .unwrap();
        let prefill = model.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(prefill.shape().dims(), &[2, config.vocab_size]);
        assert_eq!(cache.sequence_length(0).unwrap(), 2);
        let decode = model.forward(&[3], &mut cache).unwrap();
        assert_eq!(decode.shape().dims(), &[1, config.vocab_size]);
        assert_eq!(cache.sequence_length(0).unwrap(), 3);
    }

    #[test]
    fn attention_workspace_reuse_matches_fresh_scores() {
        let mut config = config();
        config.num_hidden_layers = 2;
        let model = |workspace_reuse| {
            Qwen3Model::new(
                config.clone(),
                CpuScalar::with_config(CpuRuntimeConfig {
                    workspace_reuse,
                    ..CpuScalar::new().config()
                }),
                filled(vec![config.vocab_size, config.hidden_size], 0.01),
                vec![layer(), layer()],
                filled(vec![config.hidden_size], 1.0),
                filled(vec![config.hidden_size, config.vocab_size], 0.01),
            )
            .unwrap()
        };
        let reused = model(true);
        let fresh = model(false);
        let mut reused_cache = KvCache::new(2, config.max_position_embeddings, 1, 4).unwrap();
        let mut fresh_cache = reused_cache.clone();
        for prompt in [&[1, 2][..], &[3][..], &[4][..]] {
            assert_eq!(
                reused.forward(prompt, &mut reused_cache).unwrap(),
                fresh.forward(prompt, &mut fresh_cache).unwrap()
            );
        }
        assert_eq!(fresh_cache.attention_scores.capacity(), 0);
        assert!(reused_cache.attention_scores.capacity() >= 4);
        let scratch = reused_cache.attention_scores.as_ptr();
        reused_cache.clear();
        reused.forward(&[1, 2], &mut reused_cache).unwrap();
        assert_eq!(reused_cache.attention_scores.as_ptr(), scratch);
    }

    #[test]
    fn streams_tokens_and_honors_sink_stop() {
        let config = config();
        let model = Qwen3Model::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(1, config.max_position_embeddings, 1, 4).unwrap();
        let mut streamed = Vec::new();
        let generated = generate_qwen3_stream(
            &model,
            &[1, 2],
            &mut cache,
            &GenerationConfig {
                max_new_tokens: 4,
                ..GenerationConfig::default()
            },
            &mut |token| {
                streamed.push(token);
                Ok(StreamControl::Stop)
            },
        )
        .unwrap();
        assert_eq!(generated, streamed);
        assert_eq!(streamed.len(), 1);
    }

    #[test]
    fn runs_qwen3_with_q8_linear_weights() {
        let config = config();
        let model = Qwen3Model::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![quantized_layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.effective_head_dim(),
        )
        .unwrap();
        let logits = model.forward(&[1], &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[1, config.vocab_size]);
    }

    #[test]
    fn loads_qwen3_from_gguf_with_q8_projection() {
        let config = config();
        let mut writer = GgufWriter::new();
        writer.insert_metadata(
            "general.architecture",
            GgufValue::String("qwen3".to_owned()),
        );
        writer.insert_metadata("qwen3.vocab_size", GgufValue::U32(16));
        writer.insert_metadata("qwen3.context_length", GgufValue::U32(32));
        writer.insert_metadata("qwen3.embedding_length", GgufValue::U32(8));
        writer.insert_metadata("qwen3.feed_forward_length", GgufValue::U32(16));
        writer.insert_metadata("qwen3.block_count", GgufValue::U32(1));
        writer.insert_metadata("qwen3.attention.head_count", GgufValue::U32(2));
        writer.insert_metadata("qwen3.attention.head_count_kv", GgufValue::U32(1));
        writer.insert_metadata("qwen3.attention.head_dim", GgufValue::U32(4));
        writer.insert_metadata(
            "qwen3.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-6),
        );
        writer.insert_metadata("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0));
        writer.insert_metadata("qwen3.tie_word_embeddings", GgufValue::Bool(true));
        writer.insert_metadata(
            "qwen3.architectures",
            GgufValue::Array(vec![GgufValue::String("Qwen3ForCausalLM".to_owned())]),
        );
        writer
            .add_f32_tensor("token_embd.weight", vec![16, 8], &vec![0.01; 128])
            .unwrap();
        writer
            .add_f32_tensor("output_norm.weight", vec![8], &[1.0; 8])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_norm.weight", vec![8], &[1.0; 8])
            .unwrap();
        writer.add_q8_0_tensor(
            "blk.0.attn_q.weight",
            &quantize_q8_0(&filled(vec![8, 8], 0.01)).unwrap(),
        );
        writer
            .add_f32_tensor("blk.0.attn_q_norm.weight", vec![4], &[1.0; 4])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_k.weight", vec![8, 4], &[0.01; 32])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_k_norm.weight", vec![4], &[1.0; 4])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_v.weight", vec![8, 4], &[0.01; 32])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_output.weight", vec![8, 8], &vec![0.01; 64])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.ffn_norm.weight", vec![8], &[1.0; 8])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.ffn_gate.weight", vec![8, 16], &vec![0.01; 128])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.ffn_up.weight", vec![8, 16], &vec![0.01; 128])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.ffn_down.weight", vec![16, 8], &vec![0.01; 128])
            .unwrap();

        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let model = Qwen3Model::from_gguf(CpuScalar::new(), &reader).unwrap();
        assert_eq!(model.config, config);
        assert_eq!(model.layers[0].q_proj.dtype(), DType::Q8_0);
        assert_eq!(model.lm_head.shape().dims(), &[8, 16]);
        let mut cache = KvCache::new(1, 32, 1, 4).unwrap();
        let logits = model.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[2, 16]);
    }

    #[test]
    fn loads_q4_linear_weight_from_gguf() {
        let mut writer = GgufWriter::new();
        writer.add_q4_0_tensor(
            "blk.0.attn_q.weight",
            &quantize_q4_0(&filled(vec![32, 1], 0.25)).unwrap(),
        );
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let linear = super::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
        assert_eq!(linear.dtype(), DType::Q4_0);
    }

    #[test]
    fn loads_q4_k_linear_weight_from_gguf() {
        let mut writer = GgufWriter::new();
        writer.add_q4_k_tensor(
            "blk.0.attn_q.weight",
            &quantize_q4_k(&filled(vec![256, 1], 0.25)).unwrap(),
        );
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let linear = super::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
        assert_eq!(linear.dtype(), DType::Q4_K);
    }

    #[test]
    fn loads_q5_k_linear_weight_from_gguf() {
        let mut writer = GgufWriter::new();
        writer.add_q5_k_tensor(
            "blk.0.attn_q.weight",
            &quantize_q5_k(&filled(vec![256, 1], 0.25)).unwrap(),
        );
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let linear = super::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
        assert_eq!(linear.dtype(), DType::Q5_K);
    }

    #[test]
    fn loads_q6_k_linear_weight_from_gguf() {
        let mut writer = GgufWriter::new();
        writer.add_q6_k_tensor(
            "blk.0.attn_q.weight",
            &quantize_q6_k(&filled(vec![256, 1], 0.25)).unwrap(),
        );
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let linear = super::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
        assert_eq!(linear.dtype(), DType::Q6_K);
    }
}
