//! Model-specific conversion adapters.

use std::collections::BTreeSet;

use quantinfer_format::{GgufValue, GgufWriter};
use quantinfer_model::llama::ggml_qk_channel;
use quantinfer_model::{LlamaAdapter, LlamaConfig, ModelAdapter, Qwen3Adapter, Qwen3Config};
use serde_json::Value;

use crate::common::metadata_u32;

#[derive(Debug)]
pub(crate) enum ConversionModel {
    Qwen3 {
        config: Qwen3Config,
        adapter: Qwen3Adapter,
    },
    Llama {
        config: LlamaConfig,
        adapter: LlamaAdapter,
    },
}

impl ConversionModel {
    pub(crate) fn from_config(
        config_json: &str,
        requested_architecture: Option<&str>,
    ) -> Result<Self, String> {
        let root: Value = serde_json::from_str(config_json)
            .map_err(|error| format!("cannot parse model config: {error}"))?;
        let architecture = requested_architecture
            .or_else(|| root.get("model_type").and_then(Value::as_str))
            .or_else(|| {
                root.get("architectures")
                    .and_then(Value::as_array)
                    .and_then(|architectures| {
                        architectures
                            .iter()
                            .filter_map(Value::as_str)
                            .find(|value| {
                                let value = value.to_ascii_lowercase();
                                value.contains("llama") || value.contains("qwen3")
                            })
                    })
            })
            .ok_or_else(|| {
                "model architecture is missing; pass --architecture qwen3 or --architecture llama"
                    .to_owned()
            })?
            .to_ascii_lowercase();
        match architecture.as_str() {
            value if value == "qwen3" || value == "qwen" || value.contains("qwen3") => {
                let config = Qwen3Config::from_json(config_json)
                    .map_err(|error| format!("cannot parse Qwen3 config: {error}"))?;
                config
                    .validate()
                    .map_err(|error| format!("invalid Qwen3 config: {error}"))?;
                let adapter = Qwen3Adapter::new(config.clone(), "qwen3")
                    .map_err(|error| format!("cannot build Qwen3 model adapter: {error}"))?;
                Ok(Self::Qwen3 { config, adapter })
            }
            value if value == "llama" || value == "llama2" || value.contains("llama") => {
                let config = LlamaConfig::from_json(config_json)
                    .map_err(|error| format!("cannot parse Llama config: {error}"))?;
                config
                    .validate()
                    .map_err(|error| format!("invalid Llama config: {error}"))?;
                let adapter = LlamaAdapter::new(config.clone(), "llama")
                    .map_err(|error| format!("cannot build Llama model adapter: {error}"))?;
                Ok(Self::Llama { config, adapter })
            }
            other => Err(format!(
                "unsupported model architecture {other}; use qwen3 or llama"
            )),
        }
    }

    pub(crate) fn architecture(&self) -> &'static str {
        match self {
            Self::Qwen3 { .. } => "qwen3",
            Self::Llama { .. } => "llama",
        }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen3 { config, .. } => config.vocab_size,
            Self::Llama { config, .. } => config.vocab_size,
        }
    }

    pub(crate) fn canonical_tensor_name(&self, source_name: &str) -> Option<String> {
        match self {
            Self::Qwen3 { adapter, .. } => adapter.canonical_tensor_name(source_name),
            Self::Llama { adapter, .. } => adapter.canonical_tensor_name(source_name),
        }
    }

    pub(crate) fn canonicalize_tensor_layout(
        &self,
        canonical_name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        match self {
            Self::Qwen3 { adapter, .. } => {
                adapter.canonicalize_tensor_layout(canonical_name, shape, values)
            }
            Self::Llama { adapter, .. } => {
                adapter.canonicalize_tensor_layout(canonical_name, shape, values)
            }
        }
    }

    pub(crate) fn ggml_tensor_layout(
        &self,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        match shape {
            [_] => Ok((shape.to_vec(), values)),
            &[output, input] if output != 0 && input != 0 => {
                let shape = vec![input, output];
                let Self::Llama { config, .. } = self else {
                    // Standard Qwen3 already uses the Hugging Face RoPE channel order.
                    return Ok((shape, values));
                };
                let expected_outputs = if name.ends_with(".attn_q.weight") {
                    Some(config.query_projection_size())
                } else if name.ends_with(".attn_k.weight") {
                    Some(config.key_value_projection_size())
                } else {
                    None
                };
                let Some(expected) = expected_outputs else {
                    return Ok((shape, values));
                };
                if output != expected || input != config.hidden_size {
                    return Err(format!("invalid shape for GGML RoPE projection {name}"));
                }
                let head_dim = config.effective_head_dim();
                let mut ordered = Vec::with_capacity(values.len());
                for row in 0..output {
                    let start = ggml_qk_channel(row, head_dim) * input;
                    ordered.extend_from_slice(&values[start..start + input]);
                }
                Ok((shape, ordered))
            }
            _ => Err(format!("invalid tensor shape for GGML export at {name}")),
        }
    }

    pub(crate) fn write_metadata(
        &self,
        writer: &mut GgufWriter,
        ggml_file_type: Option<u32>,
    ) -> Result<(), String> {
        let (
            prefix,
            vocab_size,
            hidden_size,
            intermediate_size,
            layers,
            heads,
            kv_heads,
            head_dim,
            rope_theta,
            rms_norm_eps,
            tie_word_embeddings,
            architectures,
        ) = match self {
            Self::Qwen3 { config, .. } => (
                "qwen3",
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.effective_head_dim(),
                config.rope_theta,
                config.rms_norm_eps,
                config.tie_word_embeddings,
                &config.architectures,
            ),
            Self::Llama { config, .. } => (
                "llama",
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads(),
                config.effective_head_dim(),
                config.rope_theta,
                config.rms_norm_eps,
                config.tie_word_embeddings,
                &config.architectures,
            ),
        };
        writer.insert_metadata("general.architecture", GgufValue::String(prefix.to_owned()));
        for (key, value) in [
            (
                "context_length",
                match self {
                    Self::Qwen3 { config, .. } => config.max_position_embeddings,
                    Self::Llama { config, .. } => config.max_position_embeddings,
                },
            ),
            ("embedding_length", hidden_size),
            ("vocab_size", vocab_size),
            ("feed_forward_length", intermediate_size),
            ("block_count", layers),
            ("attention.head_count", heads),
            ("attention.head_count_kv", kv_heads),
            ("attention.head_dim", head_dim),
        ] {
            writer.insert_metadata(
                format!("{prefix}.{key}"),
                GgufValue::U32(metadata_u32(value, key)?),
            );
        }
        writer.insert_metadata(
            format!("{prefix}.rope.freq_base"),
            GgufValue::F32(rope_theta),
        );
        writer.insert_metadata(
            format!("{prefix}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(rms_norm_eps),
        );
        if let Some(file_type) = ggml_file_type {
            writer.insert_metadata("general.type", GgufValue::String("model".to_owned()));
            writer.insert_metadata("general.file_type", GgufValue::U32(file_type));
            writer.insert_metadata("general.quantization_version", GgufValue::U32(2));
            for key in [
                "attention.key_length",
                "attention.value_length",
                "rope.dimension_count",
            ] {
                writer.insert_metadata(
                    format!("{prefix}.{key}"),
                    GgufValue::U32(metadata_u32(head_dim, key)?),
                );
            }
        } else {
            writer.insert_metadata(
                format!("{prefix}.tie_word_embeddings"),
                GgufValue::Bool(tie_word_embeddings),
            );
        }
        writer.insert_metadata(
            format!("{prefix}.architectures"),
            GgufValue::Array(
                architectures
                    .iter()
                    .cloned()
                    .map(GgufValue::String)
                    .collect(),
            ),
        );
        Ok(())
    }

    pub(crate) fn validate_required_tensors(
        &self,
        written_names: &BTreeSet<String>,
    ) -> Result<(), String> {
        let (tie_word_embeddings, layers, qk_norm) = match self {
            Self::Qwen3 { config, .. } => {
                (config.tie_word_embeddings, config.num_hidden_layers, true)
            }
            Self::Llama { config, .. } => {
                (config.tie_word_embeddings, config.num_hidden_layers, false)
            }
        };
        let mut required = vec![
            "token_embd.weight".to_owned(),
            "output_norm.weight".to_owned(),
        ];
        if !tie_word_embeddings {
            required.push("output.weight".to_owned());
        }
        for layer in 0..layers {
            required.extend([
                format!("blk.{layer}.attn_norm.weight"),
                format!("blk.{layer}.attn_q.weight"),
                format!("blk.{layer}.attn_k.weight"),
                format!("blk.{layer}.attn_v.weight"),
                format!("blk.{layer}.attn_output.weight"),
                format!("blk.{layer}.ffn_norm.weight"),
                format!("blk.{layer}.ffn_gate.weight"),
                format!("blk.{layer}.ffn_up.weight"),
                format!("blk.{layer}.ffn_down.weight"),
            ]);
            if qk_norm {
                required.extend([
                    format!("blk.{layer}.attn_q_norm.weight"),
                    format!("blk.{layer}.attn_k_norm.weight"),
                ]);
            }
        }
        let missing: Vec<_> = required
            .into_iter()
            .filter(|name| !written_names.contains(name))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "converted checkpoint is missing required {} tensors: {}",
                self.architecture(),
                missing.join(", ")
            ))
        }
    }
}
