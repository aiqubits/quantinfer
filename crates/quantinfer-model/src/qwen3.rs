use serde::{Deserialize, Serialize};

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
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err("attention heads must be divisible by key/value heads".to_owned());
        }
        if self.head_dim == Some(0) {
            return Err("head_dim must be greater than zero".to_owned());
        }
        if self.head_dim.is_none() && self.hidden_size % self.num_attention_heads != 0 {
            return Err(
                "hidden size must be divisible by attention heads when head_dim is absent"
                    .to_owned(),
            );
        }
        if self.rms_norm_eps <= 0.0 || self.rope_theta <= 0.0 {
            return Err("Qwen3 normalization epsilon and RoPE theta must be positive".to_owned());
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
}
