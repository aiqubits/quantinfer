use serde::{Deserialize, Serialize};

use crate::semantic::{
    ActivationKind, AttentionSpec, EmbeddingSpec, LayerSpec, ModelSpec, NormalizationKind,
    NormalizationSpec, PositionalEncodingSpec, TensorRole, TensorSemantic, VocabularySpec,
};
use quantinfer_core::Shape;

/// Map an adjacent-pair GGML Q/K row to the checkpoint's half-split channel.
pub fn ggml_qk_channel(row: usize, head_dim: usize) -> usize {
    let channel = row % head_dim;
    row - channel + channel / 2 + (channel % 2) * (head_dim / 2)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub architectures: Vec<String>,
}

const fn default_rms_norm_eps() -> f32 {
    1e-5
}

const fn default_rope_theta() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    pub fn query_projection_size(&self) -> usize {
        self.num_attention_heads * self.effective_head_dim()
    }

    pub fn key_value_projection_size(&self) -> usize {
        self.num_key_value_heads() * self.effective_head_dim()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads() == 0
            || self.max_position_embeddings == 0
        {
            return Err("Llama dimensions must be greater than zero".to_owned());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads())
        {
            return Err("attention heads must be divisible by key/value heads".to_owned());
        }
        if self.head_dim == Some(0) || !self.effective_head_dim().is_multiple_of(2) {
            return Err("head_dim must be positive and even for RoPE".to_owned());
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err("hidden size must be divisible by attention heads".to_owned());
        }
        if self.rms_norm_eps <= 0.0 || self.rope_theta <= 0.0 {
            return Err("normalization epsilon and RoPE theta must be positive".to_owned());
        }
        Ok(())
    }

    pub fn to_model_spec(&self, model_id: impl Into<String>) -> Result<ModelSpec, String> {
        self.validate()?;
        let head_dim = self.effective_head_dim();
        let normalization = NormalizationSpec {
            kind: NormalizationKind::RmsNorm,
            epsilon: self.rms_norm_eps,
        };
        let attention = AttentionSpec {
            num_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            head_dim,
            causal: true,
            qk_norm: false,
        };
        let layers = (0..self.num_hidden_layers)
            .map(|index| LayerSpec {
                index,
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                attention: attention.clone(),
                normalization: normalization.clone(),
                post_attention_normalization: normalization.clone(),
                activation: ActivationKind::Silu,
                positional_encoding: PositionalEncodingSpec::Rope {
                    theta: self.rope_theta,
                    dimension: head_dim,
                },
            })
            .collect();
        let mut tensor_map = vec![
            TensorSemantic {
                role: TensorRole::TokenEmbedding,
                layer: None,
                source_name: "token_embd.weight".to_owned(),
                logical_shape: Shape::from(vec![self.vocab_size, self.hidden_size]),
            },
            TensorSemantic {
                role: TensorRole::FinalNormalization,
                layer: None,
                source_name: "output_norm.weight".to_owned(),
                logical_shape: Shape::from(vec![self.hidden_size]),
            },
        ];
        if !self.tie_word_embeddings {
            tensor_map.push(TensorSemantic {
                role: TensorRole::LmHead,
                layer: None,
                source_name: "output.weight".to_owned(),
                logical_shape: Shape::from(vec![self.hidden_size, self.vocab_size]),
            });
        }
        for index in 0..self.num_hidden_layers {
            let add = |role, suffix, shape: Vec<usize>| TensorSemantic {
                role,
                layer: Some(index),
                source_name: format!("blk.{index}.{suffix}.weight"),
                logical_shape: Shape::from(shape),
            };
            tensor_map.extend([
                add(
                    TensorRole::InputNormalization,
                    "attn_norm",
                    vec![self.hidden_size],
                ),
                add(
                    TensorRole::QueryProjection,
                    "attn_q",
                    vec![self.hidden_size, self.query_projection_size()],
                ),
                add(
                    TensorRole::KeyProjection,
                    "attn_k",
                    vec![self.hidden_size, self.key_value_projection_size()],
                ),
                add(
                    TensorRole::ValueProjection,
                    "attn_v",
                    vec![self.hidden_size, self.key_value_projection_size()],
                ),
                add(
                    TensorRole::OutputProjection,
                    "attn_output",
                    vec![self.query_projection_size(), self.hidden_size],
                ),
                add(
                    TensorRole::PostAttentionNormalization,
                    "ffn_norm",
                    vec![self.hidden_size],
                ),
                add(
                    TensorRole::GateProjection,
                    "ffn_gate",
                    vec![self.hidden_size, self.intermediate_size],
                ),
                add(
                    TensorRole::UpProjection,
                    "ffn_up",
                    vec![self.hidden_size, self.intermediate_size],
                ),
                add(
                    TensorRole::DownProjection,
                    "ffn_down",
                    vec![self.intermediate_size, self.hidden_size],
                ),
            ]);
        }
        let spec = ModelSpec {
            model_id: model_id.into(),
            architecture: "llama".to_owned(),
            source_revision: None,
            vocabulary: VocabularySpec {
                size: self.vocab_size,
                bos_token_id: None,
                eos_token_id: None,
                add_bos_token: true,
            },
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            max_sequence_length: self.max_position_embeddings,
            layers,
            embedding: EmbeddingSpec {
                shape: Shape::from(vec![self.vocab_size, self.hidden_size]),
                tied_lm_head: self.tie_word_embeddings,
            },
            final_normalization: normalization,
            tensor_map,
        };
        spec.validate()?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaConfig;

    #[test]
    fn parses_gqa_llama_and_adapts_without_qk_norm() {
        let config = LlamaConfig::from_json(
            r#"{
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();
        config.validate().unwrap();
        let spec = config.to_model_spec("llama-fixture").unwrap();
        assert!(!spec.layers[0].attention.qk_norm);
        assert_eq!(spec.layers[0].attention.num_key_value_heads, 2);
        assert!(spec
            .tensor(crate::TensorRole::QueryNormalization, Some(0))
            .is_none());
    }
}
