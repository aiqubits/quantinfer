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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
        if self
            .vocabulary
            .bos_token_id
            .into_iter()
            .chain(self.vocabulary.eos_token_id)
            .any(|token| token >= self.vocabulary.size)
        {
            return Err("special token IDs must be within the vocabulary".to_owned());
        }
        let valid_normalization = |normalization: &NormalizationSpec| {
            normalization.epsilon.is_finite() && normalization.epsilon > 0.0
        };
        if !valid_normalization(&self.final_normalization)
            || self.layers.iter().any(|layer| {
                !valid_normalization(&layer.normalization)
                    || !valid_normalization(&layer.post_attention_normalization)
            })
        {
            return Err("normalization epsilon must be finite and positive".to_owned());
        }
        if self.embedding.shape != Shape::from(vec![self.vocabulary.size, self.hidden_size]) {
            return Err("embedding shape does not match vocabulary and hidden size".to_owned());
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
        let mut semantic_keys = std::collections::BTreeSet::new();
        if self.tensor_map.iter().any(|tensor| {
            tensor.logical_shape.dims().contains(&0)
                || tensor.logical_shape.num_elements().is_err()
                || !semantic_keys.insert((tensor.role, tensor.layer))
        }) {
            return Err("model tensor semantic roles must be unique and non-empty".to_owned());
        }
        let require_shape = |role: TensorRole, layer: Option<usize>, shape: Vec<usize>| {
            let tensor = self.tensor(role, layer).ok_or_else(|| {
                format!("model tensor semantic role {role:?} is missing for layer {layer:?}")
            })?;
            if tensor.logical_shape != Shape::from(shape) {
                return Err(format!(
                    "model tensor semantic role {role:?} has an invalid shape for layer {layer:?}"
                ));
            }
            Ok::<_, String>(())
        };
        require_shape(
            TensorRole::TokenEmbedding,
            None,
            vec![self.vocabulary.size, self.hidden_size],
        )?;
        require_shape(TensorRole::FinalNormalization, None, vec![self.hidden_size])?;
        let lm_head = self.tensor(TensorRole::LmHead, None);
        match (self.embedding.tied_lm_head, lm_head) {
            (true, Some(_)) => {
                return Err(
                    "tied embeddings must not declare a separate language-model head".to_owned(),
                )
            }
            (false, None) => {
                return Err("untied embeddings require a language-model head".to_owned())
            }
            (_, Some(tensor))
                if tensor.logical_shape
                    != Shape::from(vec![self.hidden_size, self.vocabulary.size]) =>
            {
                return Err("language-model head has an invalid shape".to_owned())
            }
            _ => {}
        }
        for layer in &self.layers {
            let index = Some(layer.index);
            let head_projection = layer
                .attention
                .num_heads
                .checked_mul(layer.attention.head_dim)
                .ok_or_else(|| "query projection dimensions overflow usize".to_owned())?;
            let key_value_projection = layer
                .attention
                .num_key_value_heads
                .checked_mul(layer.attention.head_dim)
                .ok_or_else(|| "key/value projection dimensions overflow usize".to_owned())?;
            for role in [
                TensorRole::InputNormalization,
                TensorRole::PostAttentionNormalization,
            ] {
                require_shape(role, index, vec![self.hidden_size])?;
            }
            require_shape(
                TensorRole::QueryProjection,
                index,
                vec![self.hidden_size, head_projection],
            )?;
            require_shape(
                TensorRole::KeyProjection,
                index,
                vec![self.hidden_size, key_value_projection],
            )?;
            require_shape(
                TensorRole::ValueProjection,
                index,
                vec![self.hidden_size, key_value_projection],
            )?;
            require_shape(
                TensorRole::OutputProjection,
                index,
                vec![head_projection, self.hidden_size],
            )?;
            require_shape(
                TensorRole::GateProjection,
                index,
                vec![self.hidden_size, self.intermediate_size],
            )?;
            require_shape(
                TensorRole::UpProjection,
                index,
                vec![self.hidden_size, self.intermediate_size],
            )?;
            require_shape(
                TensorRole::DownProjection,
                index,
                vec![self.intermediate_size, self.hidden_size],
            )?;
            match &layer.positional_encoding {
                PositionalEncodingSpec::None => {}
                PositionalEncodingSpec::Rope { theta, dimension }
                    if !theta.is_finite()
                        || *theta <= 0.0
                        || *dimension == 0
                        || !dimension.is_multiple_of(2)
                        || *dimension != layer.attention.head_dim =>
                {
                    return Err(format!(
                        "model layer {} has an invalid RoPE specification",
                        layer.index
                    ));
                }
                PositionalEncodingSpec::Rope { .. } => {}
            }
            if layer.attention.qk_norm {
                require_shape(
                    TensorRole::QueryNormalization,
                    index,
                    vec![layer.attention.head_dim],
                )?;
                require_shape(
                    TensorRole::KeyNormalization,
                    index,
                    vec![layer.attention.head_dim],
                )?;
            } else if self.tensor(TensorRole::QueryNormalization, index).is_some()
                || self.tensor(TensorRole::KeyNormalization, index).is_some()
            {
                return Err(format!(
                    "model layer {} declares Q/K normalization without qk_norm",
                    layer.index
                ));
            }
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
    use super::{ModelSpec, PositionalEncodingSpec, TensorRole, TensorSemantic};
    use crate::Qwen3Config;
    use quantinfer_core::Shape;

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

    #[test]
    fn rejects_duplicate_semantic_roles() {
        let config = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
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
        let mut spec = config.to_model_spec("fixture").unwrap();
        spec.tensor_map.push(spec.tensor_map[0].clone());
        assert!(spec.validate().is_err());
    }

    #[test]
    fn rejects_missing_or_mismatched_tensor_semantics() {
        let config = Qwen3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
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
        let mut spec = config.to_model_spec("fixture").unwrap();
        spec.tensor_map
            .retain(|tensor| tensor.role != TensorRole::KeyProjection);
        assert!(spec.validate().is_err());

        let mut spec = config.to_model_spec("fixture").unwrap();
        spec.vocabulary.eos_token_id = Some(spec.vocabulary.size);
        assert!(spec.validate().is_err());

        let mut spec = config.to_model_spec("fixture").unwrap();
        spec.embedding.tied_lm_head = false;
        assert!(spec.validate().is_err());
        spec.embedding.tied_lm_head = true;
        spec.tensor_map.push(TensorSemantic {
            role: TensorRole::LmHead,
            layer: None,
            logical_shape: Shape::from(vec![spec.hidden_size, spec.vocabulary.size]),
        });
        assert!(spec.validate().is_err());

        let mut spec = config.to_model_spec("fixture").unwrap();
        spec.layers[0].positional_encoding = PositionalEncodingSpec::Rope {
            theta: f32::NAN,
            dimension: spec.layers[0].attention.head_dim,
        };
        assert!(spec.validate().is_err());
    }
}
