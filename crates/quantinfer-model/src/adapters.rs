//! Architecture adapters that expose validated semantic model specifications.
//!
//! Serialized tensor names and physical layout transforms live at format
//! boundaries, so this module contains only architecture semantics.

use crate::{LlamaConfig, ModelSpec, Qwen3Config};

pub trait ModelAdapter {
    fn architecture(&self) -> &'static str;
    fn model_spec(&self) -> &ModelSpec;
}

#[derive(Debug, Clone)]
pub struct Qwen3Adapter {
    config: Qwen3Config,
    spec: ModelSpec,
}

impl Qwen3Adapter {
    pub fn new(config: Qwen3Config, model_id: impl Into<String>) -> Result<Self, String> {
        let spec = config.to_model_spec(model_id)?;
        Ok(Self { config, spec })
    }

    pub fn from_config(config: Qwen3Config) -> Result<Self, String> {
        Self::new(config, "qwen3")
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }
}

impl ModelAdapter for Qwen3Adapter {
    fn architecture(&self) -> &'static str {
        "qwen3"
    }

    fn model_spec(&self) -> &ModelSpec {
        &self.spec
    }
}

#[derive(Debug, Clone)]
pub struct LlamaAdapter {
    config: LlamaConfig,
    spec: ModelSpec,
}

impl LlamaAdapter {
    pub fn new(config: LlamaConfig, model_id: impl Into<String>) -> Result<Self, String> {
        let spec = config.to_model_spec(model_id)?;
        Ok(Self { config, spec })
    }

    pub fn from_config(config: LlamaConfig) -> Result<Self, String> {
        Self::new(config, "llama")
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
}

impl ModelAdapter for LlamaAdapter {
    fn architecture(&self) -> &'static str {
        "llama"
    }

    fn model_spec(&self) -> &ModelSpec {
        &self.spec
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelAdapter, Qwen3Adapter};
    use crate::Qwen3Config;

    #[test]
    fn adapters_expose_semantics_without_format_names() {
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
        let adapter = Qwen3Adapter::new(config, "fixture").unwrap();
        assert_eq!(adapter.architecture(), "qwen3");
        assert_eq!(adapter.model_spec().architecture, "qwen3");
        assert!(adapter
            .model_spec()
            .tensor(crate::TensorRole::QueryProjection, Some(0))
            .is_some());
    }
}
