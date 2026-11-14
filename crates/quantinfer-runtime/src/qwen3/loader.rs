//! GGUF loading and metadata decoding for Qwen3.

use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_model::{ModelSpec, Qwen3Adapter, Qwen3Config, TensorRole};

use crate::ggml::{load_ggml_as_float, load_ggml_linear, load_ggml_matrix};
use crate::linear::{load_linear, transpose_matrix};
use crate::storage::dense_storage_name;
use crate::{Result, RuntimeError};

use super::{expect_shape, Qwen3LayerWeights, Qwen3Model};

impl<B: quantinfer_kernel::KernelBackend> Qwen3Model<B> {
    pub fn spec_from_gguf(reader: &GgufReader) -> Result<ModelSpec> {
        Qwen3Adapter::new(
            config_from_gguf(reader, reader.uses_ggml_output_rows()?)?,
            "qwen3",
        )
        .map(|adapter| adapter.spec().clone())
        .map_err(RuntimeError::InvalidInput)
    }

    pub fn from_gguf(backend: B, reader: &GgufReader) -> Result<Self> {
        let ggml_layout = reader.uses_ggml_output_rows()?;
        let adapter = Qwen3Adapter::new(config_from_gguf(reader, ggml_layout)?, "qwen3")
            .map_err(RuntimeError::InvalidInput)?;
        let config = adapter.config().clone();
        let spec = adapter.spec().clone();
        let embedding_name = crate::tensor_source(&spec, TensorRole::TokenEmbedding, None)?;
        let embedding = if ggml_layout {
            let embedding = load_ggml_as_float(reader, &embedding_name)?;
            expect_shape(
                &embedding,
                &[config.hidden_size, config.vocab_size],
                "GGML embedding",
            )?;
            embedding.reshape(spec.embedding.shape.clone())?
        } else {
            reader.tensor_f32(&embedding_name)?
        };
        let final_norm_name = crate::tensor_source(&spec, TensorRole::FinalNormalization, None)?;
        let final_norm = reader.tensor_f32(&final_norm_name)?;
        let head_name = crate::lm_head_source(&spec)?;
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
            let name = |role| crate::tensor_source(&spec, role, Some(layer.index));
            let float = |role| Ok::<_, RuntimeError>(reader.tensor_f32(&name(role)?)?);
            let linear = |role| {
                let name = name(role)?;
                if ggml_layout {
                    load_ggml_linear(reader, &name, None)
                } else {
                    load_linear(reader, &name)
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

fn config_from_gguf(reader: &GgufReader, ggml_layout: bool) -> Result<Qwen3Config> {
    let embedding_name = dense_storage_name(TensorRole::TokenEmbedding, None);
    let embedding = reader.tensor_info(&embedding_name)?;
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
            .tensor_info(&dense_storage_name(TensorRole::GateProjection, Some(0)))
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
    let lm_head_name = dense_storage_name(TensorRole::LmHead, None);
    let tie_word_embeddings = metadata_bool(reader, "qwen3.tie_word_embeddings")?
        .unwrap_or_else(|| ggml_layout && reader.tensor_info(&lm_head_name).is_err());
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
