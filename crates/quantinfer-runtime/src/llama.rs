use quantinfer_core::{DType, Shape, Tensor};
use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::LlamaConfig;

use crate::qwen3::{add, gqa_attention, rope_tables, transpose_matrix, LinearWeight};
use crate::{KvCache, Result, RuntimeError};

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
        layer_index: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let sequence_length = hidden.shape().dims()[0];
        let residual = hidden.clone();
        let normalized = backend.rms_norm(hidden, &self.input_layernorm, config.rms_norm_eps)?;
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
        let start_position = cache.sequence_length(layer_index)?;
        let (cos, sin) = rope_tables(
            config.rope_theta,
            config.effective_head_dim(),
            start_position,
            sequence_length,
        );
        let query = backend.apply_rope(&query, &cos, &sin)?;
        let key = backend.apply_rope(&key, &cos, &sin)?;
        cache.append(layer_index, &key, &value)?;
        let attended = gqa_attention(
            &query,
            cache.key_values(layer_index)?,
            cache.value_values(layer_index)?,
            cache.sequence_length(layer_index)?,
            config.num_attention_heads,
            config.num_key_value_heads(),
            config.effective_head_dim(),
            start_position,
        )?
        .reshape(Shape::from(vec![
            sequence_length,
            config.query_projection_size(),
        ]))?;
        let projected = self.o_proj.matmul(backend, &attended)?;
        let residual = add(&residual, &projected)?;
        let normalized = backend.rms_norm(
            &residual,
            &self.post_attention_layernorm,
            config.rms_norm_eps,
        )?;
        let gate = self.gate_proj.matmul(backend, &normalized)?;
        let up = self.up_proj.matmul(backend, &normalized)?;
        let feed_forward = backend.silu_mul(&gate, &up)?;
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
}

impl<B: KernelBackend> LlamaModel<B> {
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
        })
    }

    pub fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
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
        let mut hidden = self.backend.embedding(&self.embedding, token_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&self.backend, &self.config, &hidden, index, cache)?;
        }
        let hidden = self
            .backend
            .rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        Ok(self.backend.matmul(&hidden, &self.lm_head)?)
    }

    pub fn from_gguf(backend: B, reader: &GgufReader) -> Result<Self> {
        let config = config_from_gguf(reader)?;
        let embedding = reader.tensor_f32("token_embd.weight")?;
        let final_norm = reader.tensor_f32("output_norm.weight")?;
        let lm_head = match reader.tensor_info("output.weight") {
            Ok(info) if matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) => {
                reader.tensor_f32("output.weight")?
            }
            Ok(_) => {
                return Err(RuntimeError::InvalidInput(
                    "Llama output.weight must be floating point".to_owned(),
                ))
            }
            Err(_) if config.tie_word_embeddings => transpose_matrix(&embedding)?,
            Err(error) => return Err(error.into()),
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let tensor = |suffix: &str| format!("blk.{layer_index}.{suffix}.weight");
            layers.push(LlamaLayerWeights {
                input_layernorm: reader.tensor_f32(tensor("attn_norm").as_str())?,
                q_proj: load_linear(reader, tensor("attn_q").as_str())?,
                k_proj: load_linear(reader, tensor("attn_k").as_str())?,
                v_proj: load_linear(reader, tensor("attn_v").as_str())?,
                o_proj: load_linear(reader, tensor("attn_output").as_str())?,
                post_attention_layernorm: reader.tensor_f32(tensor("ffn_norm").as_str())?,
                gate_proj: load_linear(reader, tensor("ffn_gate").as_str())?,
                up_proj: load_linear(reader, tensor("ffn_up").as_str())?,
                down_proj: load_linear(reader, tensor("ffn_down").as_str())?,
            });
        }
        Self::new(config, backend, embedding, layers, final_norm, lm_head)
    }
}

fn load_linear(reader: &GgufReader, name: &str) -> Result<LinearWeight> {
    match reader.tensor_info(name)?.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            Ok(LinearWeight::F32(reader.tensor_f32(name)?))
        }
        GgmlType::Q8_0 => Ok(LinearWeight::Q8_0(reader.tensor_q8_0(name)?)),
        GgmlType::Q4_0 => Ok(LinearWeight::Q4_0(reader.tensor_q4_0(name)?)),
        GgmlType::Q4_K => Ok(LinearWeight::Q4_K(reader.tensor_q4_k(name)?)),
        GgmlType::Q5_K | GgmlType::Q6_K => Err(RuntimeError::InvalidInput(format!(
            "GGUF tensor {name} uses an unsupported K-quant runtime type"
        ))),
    }
}

fn config_from_gguf(reader: &GgufReader) -> Result<LlamaConfig> {
    let embedding = reader.tensor_info("token_embd.weight")?;
    let vocab_size = metadata_usize(reader, "llama.vocab_size")?
        .unwrap_or_else(|| embedding.shape.first().copied().unwrap_or_default());
    let hidden_size = metadata_usize(reader, "llama.embedding_length")?
        .unwrap_or_else(|| embedding.shape.get(1).copied().unwrap_or_default());
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
    let head_dim = metadata_usize(reader, "llama.attention.head_dim")?;
    let rms_norm_eps =
        metadata_f32(reader, "llama.attention.layer_norm_rms_epsilon")?.unwrap_or(1e-5);
    let rope_theta = metadata_f32(reader, "llama.rope.freq_base")?.unwrap_or(10_000.0);
    let tie_word_embeddings = metadata_bool(reader, "llama.tie_word_embeddings")?.unwrap_or(false);
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
    use crate::KvCache;
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
    }
}
