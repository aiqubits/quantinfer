use crate::{LlamaConfig, ModelSpec, Qwen3Config};

pub trait ModelAdapter {
    fn architecture(&self) -> &'static str;
    fn model_spec(&self) -> &ModelSpec;
    fn canonical_tensor_name(&self, source_name: &str) -> Option<String>;
    fn canonicalize_tensor_layout(
        &self,
        canonical_name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String>;
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

    fn canonical_tensor_name(&self, source_name: &str) -> Option<String> {
        canonical_qwen3_tensor_name(source_name)
    }

    fn canonicalize_tensor_layout(
        &self,
        canonical_name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        canonicalize_layout(canonical_name, shape, values)
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

    fn canonical_tensor_name(&self, source_name: &str) -> Option<String> {
        canonical_llama_tensor_name(source_name)
    }

    fn canonicalize_tensor_layout(
        &self,
        canonical_name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        canonicalize_layout(canonical_name, shape, values)
    }
}

fn canonical_qwen3_tensor_name(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.as_slice() {
        ["model", "embed_tokens", "weight"] | ["model", "token_embeddings", "weight"] => {
            Some("token_embd.weight".to_owned())
        }
        ["model", "norm", "weight"] => Some("output_norm.weight".to_owned()),
        ["lm_head", "weight"] => Some("output.weight".to_owned()),
        ["model", "layers", index, "input_layernorm", "weight"] => {
            Some(format!("blk.{index}.attn_norm.weight"))
        }
        ["model", "layers", index, "self_attn", projection, "weight"] => {
            let suffix = match *projection {
                "q_proj" => "attn_q",
                "q_norm" => "attn_q_norm",
                "k_proj" => "attn_k",
                "k_norm" => "attn_k_norm",
                "v_proj" => "attn_v",
                "o_proj" => "attn_output",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        ["model", "layers", index, "post_attention_layernorm", "weight"] => {
            Some(format!("blk.{index}.ffn_norm.weight"))
        }
        ["model", "layers", index, "mlp", projection, "weight"] => {
            let suffix = match *projection {
                "gate_proj" => "ffn_gate",
                "up_proj" => "ffn_up",
                "down_proj" => "ffn_down",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        _ => None,
    }
}

fn canonical_llama_tensor_name(name: &str) -> Option<String> {
    canonical_qwen3_tensor_name(name)
}

fn canonicalize_layout(
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if !is_matrix_weight(name) {
        return Ok((shape.to_vec(), values));
    }
    if shape.len() != 2 {
        return Err(format!("linear tensor {name} must be rank 2"));
    }
    let rows = shape[0];
    let columns = shape[1];
    if values.len() != rows * columns {
        return Err(format!("linear tensor {name} shape does not match data"));
    }
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }
    Ok((vec![columns, rows], transposed))
}

fn is_matrix_weight(name: &str) -> bool {
    name == "output.weight"
        || [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::{ModelAdapter, Qwen3Adapter};
    use crate::Qwen3Config;

    #[test]
    fn keeps_format_names_out_of_runtime_semantics() {
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
        assert_eq!(
            adapter.canonical_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".to_owned())
        );
        let (shape, values) = adapter
            .canonicalize_tensor_layout("blk.0.attn_q.weight", &[2, 3], vec![1.0; 6])
            .unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(values, vec![1.0; 6]);
    }
}
