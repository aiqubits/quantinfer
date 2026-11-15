use serde::{Deserialize, Serialize};

use quantinfer_core::Shape;

use crate::semantic::{
    ActivationKind, AttentionSpec, EmbeddingSpec, LayerSpec, ModelSpec, NormalizationKind,
    NormalizationSpec, PositionalEncodingSpec, TensorRole, TensorSemantic, VocabularySpec,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
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
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub architectures: Vec<String>,
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

impl Qwen3Config {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.max_position_embeddings == 0
        {
            return Err("Qwen3 dimensions must be greater than zero".to_owned());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err("attention heads must be divisible by key/value heads".to_owned());
        }
        if self.head_dim == Some(0) {
            return Err("head_dim must be greater than zero".to_owned());
        }
        if !self.effective_head_dim().is_multiple_of(2) {
            return Err("head_dim must be even for RoPE".to_owned());
        }
        if self.head_dim.is_none() && !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(
                "hidden size must be divisible by attention heads when head_dim is absent"
                    .to_owned(),
            );
        }
        if !self.rms_norm_eps.is_finite()
            || !self.rope_theta.is_finite()
            || self.rms_norm_eps <= 0.0
            || self.rope_theta <= 0.0
        {
            return Err("Qwen3 normalization epsilon and RoPE theta must be positive".to_owned());
        }
        if self
            .num_attention_heads
            .checked_mul(self.effective_head_dim())
            .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.effective_head_dim())
                .is_none()
        {
            return Err("Qwen3 projection dimensions overflow usize".to_owned());
        }
        if !self.architectures.is_empty()
            && !self
                .architectures
                .iter()
                .any(|architecture| architecture == "Qwen3ForCausalLM")
        {
            return Err("Qwen3 config does not declare Qwen3ForCausalLM".to_owned());
        }
        if self.use_sliding_window || self.sliding_window.is_some() {
            return Err("sliding-window attention is not implemented in the first MVP".to_owned());
        }
        if self.rope_scaling.is_some() {
            return Err("custom RoPE scaling is not implemented in the first MVP".to_owned());
        }
        Ok(())
    }

    pub fn query_projection_size(&self) -> usize {
        self.num_attention_heads * self.effective_head_dim()
    }

    pub fn key_value_projection_size(&self) -> usize {
        self.num_key_value_heads * self.effective_head_dim()
    }

    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
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
                logical_shape: Shape::from(vec![self.vocab_size, self.hidden_size]),
            },
            TensorSemantic {
                role: TensorRole::FinalNormalization,
                layer: None,
                logical_shape: Shape::from(vec![self.hidden_size]),
            },
        ];
        if !self.tie_word_embeddings {
            tensor_map.push(TensorSemantic {
                role: TensorRole::LmHead,
                layer: None,
                logical_shape: Shape::from(vec![self.hidden_size, self.vocab_size]),
            });
        }
        for index in 0..self.num_hidden_layers {
            let add = |role, shape: Vec<usize>| TensorSemantic {
                role,
                layer: Some(index),
                logical_shape: Shape::from(shape),
            };
            tensor_map.extend([
                add(TensorRole::InputNormalization, vec![self.hidden_size]),
                add(
                    TensorRole::QueryProjection,
                    vec![self.hidden_size, self.query_projection_size()],
                ),
                add(TensorRole::QueryNormalization, vec![head_dim]),
                add(
                    TensorRole::KeyProjection,
                    vec![self.hidden_size, self.key_value_projection_size()],
                ),
                add(TensorRole::KeyNormalization, vec![head_dim]),
                add(
                    TensorRole::ValueProjection,
                    vec![self.hidden_size, self.key_value_projection_size()],
                ),
                add(
                    TensorRole::OutputProjection,
                    vec![self.query_projection_size(), self.hidden_size],
                ),
                add(
                    TensorRole::PostAttentionNormalization,
                    vec![self.hidden_size],
                ),
                add(
                    TensorRole::GateProjection,
                    vec![self.hidden_size, self.intermediate_size],
                ),
                add(
                    TensorRole::UpProjection,
                    vec![self.hidden_size, self.intermediate_size],
                ),
                add(
                    TensorRole::DownProjection,
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
    use super::Qwen3Config;

    #[test]
    fn parses_and_validates_dense_qwen3_config() {
        let config = Qwen3Config::from_json(
            r#"{
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true,
                "use_sliding_window": false
            }"#,
        )
        .unwrap();

        assert_eq!(config.query_projection_size(), 64);
        assert_eq!(config.key_value_projection_size(), 32);
        assert_eq!(config.kv_group_size(), 2);
        config.validate().unwrap();
    }

    #[test]
    fn rejects_sliding_window_configs() {
        let mut config = Qwen3Config {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(16),
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            sliding_window: Some(64),
            max_window_layers: Some(2),
            use_sliding_window: true,
            rope_scaling: None,
            architectures: Vec::new(),
        };
        assert!(config.validate().is_err());
        config.sliding_window = None;
        config.use_sliding_window = false;
        config.validate().unwrap();
    }

    #[test]
    fn rejects_non_finite_and_overflowing_dimensions() {
        let mut config = Qwen3Config {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: Some(usize::MAX - 1),
            rms_norm_eps: f32::NAN,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: false,
            rope_scaling: None,
            architectures: Vec::new(),
        };
        assert!(config.validate().is_err());
        config.rms_norm_eps = 1e-6;
        config.head_dim = Some(usize::MAX - 1);
        assert!(config.validate().is_err());
    }
}
