use quantinfer_core::{
    DType, QuantizedTensor, QuantizedTensor4, Shape, Tensor, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::llama::ggml_qk_channel;
use quantinfer_model::{LlamaAdapter, LlamaConfig, ModelSpec, TensorRole};
use quantinfer_quant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};

use crate::batch::{last_hidden_rows, DenseAttention, DenseBatchLimits};
use crate::calibration::{ProjectionObserver, ProjectionTrace};
use crate::qwen3::{add, load_linear, transpose_matrix, LinearWeight};
use crate::{DecodeBatchItem, KvCache, LoraAdapter, LoraProjection, Result, RuntimeError};

#[derive(Debug, Clone)]
pub struct LlamaLayerWeights {
    pub input_layernorm: Tensor,
    pub q_proj: LinearWeight,
    pub k_proj: LinearWeight,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub post_attention_layernorm: Tensor,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

impl LlamaLayerWeights {
    pub fn validate(&self, config: &LlamaConfig) -> Result<()> {
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
                "k_proj",
                self.k_proj.shape().dims(),
                &[hidden, key_value][..],
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
                    "Llama {name} has shape {actual:?}, expected {expected:?}"
                )));
            }
            if !matches!(
                name,
                "q_proj" | "k_proj" | "v_proj" | "o_proj" | "gate_proj" | "up_proj" | "down_proj"
            ) && self.weight_dtype(name)? != DType::F32
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "Llama {name} must use F32 for the reference runtime"
                )));
            }
        }
        Ok(())
    }

    fn weight_dtype(&self, name: &str) -> Result<DType> {
        match name {
            "input_layernorm" => Ok(self.input_layernorm.dtype()),
            "q_proj" => Ok(self.q_proj.dtype()),
            "k_proj" => Ok(self.k_proj.dtype()),
            "v_proj" => Ok(self.v_proj.dtype()),
            "o_proj" => Ok(self.o_proj.dtype()),
            "post_attention_layernorm" => Ok(self.post_attention_layernorm.dtype()),
            "gate_proj" => Ok(self.gate_proj.dtype()),
            "up_proj" => Ok(self.up_proj.dtype()),
            "down_proj" => Ok(self.down_proj.dtype()),
            _ => Err(RuntimeError::InvalidInput(format!(
                "unknown Llama weight {name}"
            ))),
        }
    }

    fn forward<B: KernelBackend>(
        &self,
        backend: &B,
        config: &LlamaConfig,
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
                config.num_key_value_heads(),
                config.effective_head_dim(),
            ]))?;
        let value = self
            .v_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads(),
                config.effective_head_dim(),
            ]))?;
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

pub struct LlamaModel<B> {
    pub config: LlamaConfig,
    backend: B,
    pub embedding: Tensor,
    pub layers: Vec<LlamaLayerWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
    lora_head: Option<LoraProjection>,
}

impl<B: KernelBackend> LlamaModel<B> {
    pub fn spec_from_gguf(reader: &GgufReader) -> Result<ModelSpec> {
        config_from_gguf(reader, uses_ggml_layout(reader)?)?
            .to_model_spec("llama")
            .map_err(RuntimeError::InvalidInput)
    }

    pub fn new(
        config: LlamaConfig,
        backend: B,
        embedding: Tensor,
        layers: Vec<LlamaLayerWeights>,
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
                "Llama has {} layers but received {} weight sets",
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
                "Llama forward requires at least one token".to_owned(),
            ));
        }
        if cache.sequence_length(0)? + token_ids.len() > self.config.max_position_embeddings {
            return Err(RuntimeError::InvalidInput(
                "input exceeds Llama max_position_embeddings".to_owned(),
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
            key_value_heads: self.config.num_key_value_heads(),
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
            key_value_heads: self.config.num_key_value_heads(),
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
        let ggml_layout = uses_ggml_layout(reader)?;
        let config = config_from_gguf(reader, ggml_layout)?;
        let adapter =
            LlamaAdapter::from_config(config.clone()).map_err(RuntimeError::InvalidInput)?;
        let spec = adapter.spec();
        let embedding_name = crate::tensor_source(spec, TensorRole::TokenEmbedding, None)?;
        let mut embedding = if ggml_layout {
            load_ggml_as_float(reader, embedding_name)?
        } else {
            reader.tensor_f32(embedding_name)?
        };
        if ggml_layout {
            expect_shape(
                &embedding,
                &[config.hidden_size, config.vocab_size],
                "GGML embedding",
            )?;
            embedding = embedding.reshape(spec.embedding.shape.clone())?;
        }
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
            Ok(_) => {
                return Err(RuntimeError::InvalidInput(format!(
                    "Llama LM head {head_name} must be floating point"
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
                if !ggml_layout {
                    return load_linear(reader, name);
                }
                let heads = match role {
                    TensorRole::QueryProjection => Some(config.num_attention_heads),
                    TensorRole::KeyProjection => Some(config.num_key_value_heads()),
                    _ => None,
                };
                load_ggml_linear(reader, name, heads)
            };
            layers.push(LlamaLayerWeights {
                input_layernorm: float(TensorRole::InputNormalization)?,
                q_proj: linear(TensorRole::QueryProjection)?,
                k_proj: linear(TensorRole::KeyProjection)?,
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

pub fn uses_ggml_layout(reader: &GgufReader) -> Result<bool> {
    Ok(reader.llama_uses_ggml_layout()?)
}

fn load_ggml_as_float(reader: &GgufReader, name: &str) -> Result<Tensor> {
    match reader.tensor_info(name)?.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => return Ok(reader.tensor_f32(name)?),
        GgmlType::Q8_0 => dequantize_q8_0(&reader.tensor_q8_0(name)?),
        GgmlType::Q4_0 => dequantize_q4_0(&reader.tensor_q4_0(name)?),
        GgmlType::Q4_K => dequantize_q4_k(&reader.tensor_q4_k(name)?),
        GgmlType::Q5_K => dequantize_q5_k(&reader.tensor_q5_k(name)?),
        GgmlType::Q6_K => dequantize_q6_k(&reader.tensor_q6_k(name)?),
    }
    .map_err(|error| {
        RuntimeError::InvalidInput(format!("cannot decode GGML tensor {name}: {error}"))
    })
}

fn load_ggml_linear(
    reader: &GgufReader,
    name: &str,
    rope_heads: Option<usize>,
) -> Result<LinearWeight> {
    let info = reader.tensor_info(name)?;
    let &[input, output] = info.shape.as_slice() else {
        return Err(RuntimeError::InvalidInput(format!(
            "GGML matrix {name} must be rank 2"
        )));
    };
    match info.dtype {
        GgmlType::Q8_0 if input != 0 && input.is_multiple_of(Q8_0_BLOCK_SIZE) => {
            let weight = reader.tensor_q8_0(name)?;
            let weight = if let Some(dim) = rope_head_dim(name, output, rope_heads)? {
                QuantizedTensor::from_blocks(
                    weight.shape().clone(),
                    reorder_ggml_rows(weight.blocks(), input / Q8_0_BLOCK_SIZE, dim),
                )?
            } else {
                weight
            };
            Ok(LinearWeight::Q8_0OutputRows(weight))
        }
        GgmlType::Q4_0 if input != 0 && input.is_multiple_of(Q4_0_BLOCK_SIZE) => {
            let weight = reader.tensor_q4_0(name)?;
            let weight = if let Some(dim) = rope_head_dim(name, output, rope_heads)? {
                QuantizedTensor4::from_blocks(
                    weight.shape().clone(),
                    reorder_ggml_rows(weight.blocks(), input / Q4_0_BLOCK_SIZE, dim),
                )?
            } else {
                weight
            };
            Ok(LinearWeight::Q4_0OutputRows(weight))
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Ok(LinearWeight::F32(load_ggml_matrix(
            reader, name, rope_heads,
        )?)),
        dtype => Err(RuntimeError::InvalidInput(format!(
            "unsupported GGML projection {name} with input width {input} and type {dtype:?}"
        ))),
    }
}

fn rope_head_dim(name: &str, output: usize, rope_heads: Option<usize>) -> Result<Option<usize>> {
    rope_heads
        .map(|heads| {
            if heads == 0
                || output == 0
                || !output.is_multiple_of(heads)
                || !(output / heads).is_multiple_of(2)
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "GGML matrix {name} has an invalid RoPE layout"
                )));
            }
            Ok(output / heads)
        })
        .transpose()
}

fn reorder_ggml_rows<T: Clone>(blocks: &[T], row_blocks: usize, dim: usize) -> Vec<T> {
    let mut ordered = blocks.to_vec();
    for (row, values) in blocks.chunks_exact(row_blocks).enumerate() {
        let start = ggml_qk_channel(row, dim) * row_blocks;
        ordered[start..start + row_blocks].clone_from_slice(values);
    }
    ordered
}

fn load_ggml_matrix(reader: &GgufReader, name: &str, rope_heads: Option<usize>) -> Result<Tensor> {
    let tensor = load_ggml_as_float(reader, name)?;
    let &[input, output] = tensor.shape().dims() else {
        return Err(RuntimeError::InvalidInput(format!(
            "GGML matrix {name} must be rank 2"
        )));
    };
    let source = tensor.as_f32()?;
    let mut weights = vec![0.0; source.len()];
    let head_dim = rope_head_dim(name, output, rope_heads)?;
    for (output_index, row) in source.chunks_exact(input).enumerate() {
        let column = head_dim.map_or(output_index, |dim| ggml_qk_channel(output_index, dim));
        for (input_index, &value) in row.iter().enumerate() {
            weights[input_index * output + column] = value;
        }
    }
    Ok(Tensor::from_f32(vec![input, output], weights)?)
}

fn config_from_gguf(reader: &GgufReader, ggml_layout: bool) -> Result<LlamaConfig> {
    let embedding = reader.tensor_info("token_embd.weight")?;
    let (hidden_axis, vocab_axis) = if ggml_layout { (0, 1) } else { (1, 0) };
    let vocab_size = metadata_usize(reader, "llama.vocab_size")?
        .unwrap_or_else(|| embedding.shape.get(vocab_axis).copied().unwrap_or_default());
    let hidden_size = metadata_usize(reader, "llama.embedding_length")?.unwrap_or_else(|| {
        embedding
            .shape
            .get(hidden_axis)
            .copied()
            .unwrap_or_default()
    });
    let num_hidden_layers = required_metadata_usize(reader, "llama.block_count")?;
    let num_attention_heads = required_metadata_usize(reader, "llama.attention.head_count")?;
    let num_key_value_heads = metadata_usize(reader, "llama.attention.head_count_kv")?;
    let intermediate_size = metadata_usize(reader, "llama.feed_forward_length")?.or_else(|| {
        reader
            .tensor_info("blk.0.ffn_gate.weight")
            .ok()
            .and_then(|info| info.shape.get(1).copied())
    });
    let intermediate_size = intermediate_size.ok_or_else(|| {
        RuntimeError::InvalidInput("GGUF is missing llama.feed_forward_length".to_owned())
    })?;
    let max_position_embeddings = metadata_usize(reader, "llama.context_length")?.unwrap_or(2048);
    let mut head_dim = metadata_usize(reader, "llama.attention.head_dim")?;
    if ggml_layout && head_dim.is_none() {
        head_dim = metadata_usize(reader, "llama.attention.key_length")?;
    }
    let rms_norm_eps =
        metadata_f32(reader, "llama.attention.layer_norm_rms_epsilon")?.unwrap_or(1e-5);
    let rope_theta = metadata_f32(reader, "llama.rope.freq_base")?.unwrap_or(10_000.0);
    let tie_word_embeddings = metadata_bool(reader, "llama.tie_word_embeddings")?
        .unwrap_or_else(|| ggml_layout && reader.tensor_info("output.weight").is_err());
    let config = LlamaConfig {
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
        architectures: Vec::new(),
    };
    config.validate().map_err(RuntimeError::InvalidInput)?;
    if ggml_layout {
        for key in ["llama.rope.dimension_count", "llama.attention.value_length"] {
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
            .get("llama.rope.scaling.type")
            .is_some_and(|value| value != &GgufValue::String("none".to_owned()))
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

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected || tensor.dtype() != DType::F32 {
        return Err(RuntimeError::InvalidInput(format!(
            "Llama {name} has shape {:?} and dtype {:?}, expected shape {expected:?} F32",
            tensor.shape().dims(),
            tensor.dtype()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LlamaLayerWeights, LlamaModel};
    use crate::{generate_stream, GenerationConfig, KvCache, StreamControl};
    use quantinfer_backend_cpu::CpuScalar;
    use quantinfer_core::Tensor;
    use quantinfer_model::LlamaConfig;

    fn filled(shape: Vec<usize>, value: f32) -> Tensor {
        Tensor::from_f32(shape.clone(), vec![value; shape.iter().product()]).unwrap()
    }

    fn config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(1),
            head_dim: Some(4),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
            architectures: vec!["LlamaForCausalLM".to_owned()],
        }
    }

    #[test]
    fn runs_dense_llama_prefill_and_decode_with_gqa() {
        let config = config();
        let model = LlamaModel::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![LlamaLayerWeights {
                input_layernorm: filled(vec![config.hidden_size], 1.0),
                q_proj: super::LinearWeight::F32(filled(vec![8, 8], 0.01)),
                k_proj: super::LinearWeight::F32(filled(vec![8, 4], 0.01)),
                v_proj: super::LinearWeight::F32(filled(vec![8, 4], 0.01)),
                o_proj: super::LinearWeight::F32(filled(vec![8, 8], 0.01)),
                post_attention_layernorm: filled(vec![config.hidden_size], 1.0),
                gate_proj: super::LinearWeight::F32(filled(vec![8, 16], 0.01)),
                up_proj: super::LinearWeight::F32(filled(vec![8, 16], 0.01)),
                down_proj: super::LinearWeight::F32(filled(vec![16, 8], 0.01)),
            }],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(1, 32, 1, 4).unwrap();
        let prefill = model.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(prefill.shape().dims(), &[2, config.vocab_size]);
        let decode = model.forward(&[3], &mut cache).unwrap();
        assert_eq!(decode.shape().dims(), &[1, config.vocab_size]);
        assert_eq!(cache.sequence_length(0).unwrap(), 3);

        cache.clear();
        let mut emitted = Vec::new();
        let generated = generate_stream(
            &model,
            &[1, 2],
            &mut cache,
            &GenerationConfig {
                max_new_tokens: 4,
                eos_token_id: Some(0),
                ..GenerationConfig::default()
            },
            &mut |token| {
                emitted.push(token);
                Ok(StreamControl::Continue)
            },
        )
        .unwrap();
        assert_eq!(emitted, [0]);
        assert_eq!(generated, emitted);
        assert_eq!(cache.sequence_length(0).unwrap(), 2);
    }
}
