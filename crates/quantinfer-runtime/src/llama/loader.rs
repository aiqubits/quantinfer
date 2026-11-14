//! GGUF loading and metadata decoding for Llama.

use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_model::{LlamaAdapter, LlamaConfig, ModelSpec, TensorRole};

use crate::ggml::{load_ggml_as_float, load_ggml_linear, load_ggml_matrix};
use crate::linear::{load_linear, transpose_matrix};
use crate::storage::dense_storage_name;
use crate::{Result, RuntimeError};

use super::{expect_shape, LlamaLayerWeights, LlamaModel};

impl<B: quantinfer_kernel::KernelBackend> LlamaModel<B> {
    pub fn spec_from_gguf(reader: &GgufReader) -> Result<ModelSpec> {
        LlamaAdapter::new(
            config_from_gguf(reader, uses_ggml_layout(reader)?)?,
            "llama",
        )
        .map(|adapter| adapter.spec().clone())
        .map_err(RuntimeError::InvalidInput)
    }

    pub fn from_gguf(backend: B, reader: &GgufReader) -> Result<Self> {
        let ggml_layout = uses_ggml_layout(reader)?;
        let adapter = LlamaAdapter::new(config_from_gguf(reader, ggml_layout)?, "llama")
            .map_err(RuntimeError::InvalidInput)?;
        let config = adapter.config().clone();
        let spec = adapter.spec().clone();
        let embedding_name = crate::tensor_source(&spec, TensorRole::TokenEmbedding, None)?;
        let mut embedding = if ggml_layout {
            load_ggml_as_float(reader, &embedding_name)?
        } else {
            reader.tensor_f32(&embedding_name)?
        };
        if ggml_layout {
            expect_shape(
                &embedding,
                &[config.hidden_size, config.vocab_size],
                "GGML embedding",
            )?;
            embedding = embedding.reshape(spec.embedding.shape.clone())?;
        }
        let final_norm_name = crate::tensor_source(&spec, TensorRole::FinalNormalization, None)?;
        let final_norm = reader.tensor_f32(&final_norm_name)?;
        let head_name = crate::lm_head_source(&spec)?;
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
            let name = |role| crate::tensor_source(&spec, role, Some(layer.index));
            let float = |role| Ok::<_, RuntimeError>(reader.tensor_f32(&name(role)?)?);
            let linear = |role| {
                let name = name(role)?;
                if !ggml_layout {
                    return load_linear(reader, &name);
                }
                let heads = match role {
                    TensorRole::QueryProjection => Some(config.num_attention_heads),
                    TensorRole::KeyProjection => Some(config.num_key_value_heads()),
                    _ => None,
                };
                load_ggml_linear(reader, &name, heads)
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
    Ok(reader.uses_ggml_output_rows()?)
}

fn config_from_gguf(reader: &GgufReader, ggml_layout: bool) -> Result<LlamaConfig> {
    let embedding_name = dense_storage_name(TensorRole::TokenEmbedding, None);
    let embedding = reader.tensor_info(&embedding_name)?;
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
            .tensor_info(&dense_storage_name(TensorRole::GateProjection, Some(0)))
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
    let lm_head_name = dense_storage_name(TensorRole::LmHead, None);
    let tie_word_embeddings = metadata_bool(reader, "llama.tie_word_embeddings")?
        .unwrap_or_else(|| ggml_layout && reader.tensor_info(&lm_head_name).is_err());
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
