use serde::{Deserialize, Serialize};

use quantinfer_core::Shape;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocabularySpec {
    pub size: usize,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub add_bos_token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationKind {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizationSpec {
    pub kind: NormalizationKind,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationKind {
    Silu,
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PositionalEncodingSpec {
    None,
    Rope { theta: f32, dimension: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionSpec {
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub qk_norm: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub index: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention: AttentionSpec,
    pub normalization: NormalizationSpec,
    pub post_attention_normalization: NormalizationSpec,
    pub activation: ActivationKind,
    pub positional_encoding: PositionalEncodingSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpec {
    pub shape: Shape,
    pub tied_lm_head: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    TokenEmbedding,
    FinalNormalization,
    LmHead,
    InputNormalization,
    QueryProjection,
    QueryNormalization,
    KeyProjection,
    KeyNormalization,
    ValueProjection,
    OutputProjection,
    PostAttentionNormalization,
    GateProjection,
    UpProjection,
    DownProjection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSemantic {
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub source_name: String,
    pub logical_shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub architecture: String,
    pub source_revision: Option<String>,
    pub vocabulary: VocabularySpec,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_sequence_length: usize,
    pub layers: Vec<LayerSpec>,
    pub embedding: EmbeddingSpec,
    pub final_normalization: NormalizationSpec,
    pub tensor_map: Vec<TensorSemantic>,
}

impl ModelSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.is_empty() || self.architecture.is_empty() {
            return Err("model id and architecture must not be empty".to_owned());
        }
        if self.vocabulary.size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.max_sequence_length == 0
            || self.layers.is_empty()
        {
            return Err("model dimensions must be greater than zero".to_owned());
        }
        if self.layers.iter().enumerate().any(|(index, layer)| {
            layer.index != index
                || layer.hidden_size != self.hidden_size
                || layer.intermediate_size != self.intermediate_size
                || layer.attention.num_heads == 0
                || layer.attention.num_key_value_heads == 0
                || layer.attention.head_dim == 0
                || !layer
                    .attention
                    .num_heads
                    .is_multiple_of(layer.attention.num_key_value_heads)
        }) {
            return Err("model layer specifications are inconsistent".to_owned());
        }
        Ok(())
    }

    pub fn tensor(&self, role: TensorRole, layer: Option<usize>) -> Option<&TensorSemantic> {
        self.tensor_map
            .iter()
            .find(|tensor| tensor.role == role && tensor.layer == layer)
    }
}

impl super::Qwen3Config {
    pub fn to_model_spec(&self, model_id: impl Into<String>) -> Result<ModelSpec, String> {
        self.validate()?;
        let head_dim = self.effective_head_dim();
        let normalization = NormalizationSpec {
            kind: NormalizationKind::RmsNorm,
            epsilon: self.rms_norm_eps,
        };
        let attention = AttentionSpec {
            num_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim,
            causal: true,
            qk_norm: true,
        };
        let layer_specs = (0..self.num_hidden_layers)
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
            .collect::<Vec<_>>();
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
                    TensorRole::QueryNormalization,
                    "attn_q_norm",
                    vec![head_dim],
                ),
                add(
                    TensorRole::KeyProjection,
                    "attn_k",
                    vec![self.hidden_size, self.key_value_projection_size()],
                ),
                add(TensorRole::KeyNormalization, "attn_k_norm", vec![head_dim]),
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
            architecture: "qwen3".to_owned(),
            source_revision: None,
            vocabulary: VocabularySpec {
                size: self.vocab_size,
                bos_token_id: None,
                eos_token_id: None,
                add_bos_token: false,
            },
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            max_sequence_length: self.max_position_embeddings,
            layers: layer_specs,
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
    use super::{ModelSpec, TensorRole};
    use crate::Qwen3Config;

    #[test]
    fn adapts_qwen3_to_model_semantic_ir() {
        let config = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(4),
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: false,
            rope_scaling: None,
            architectures: vec!["Qwen3ForCausalLM".to_owned()],
        };
        let spec = config.to_model_spec("fixture").unwrap();
        assert_eq!(spec.layers.len(), 2);
        assert!(spec.tensor(TensorRole::QueryProjection, Some(1)).is_some());
        assert!(spec.tensor(TensorRole::LmHead, None).is_none());
        spec.validate().unwrap();
        let encoded = serde_json::to_string(&spec).unwrap();
        let decoded: ModelSpec = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, spec);
    }
}
