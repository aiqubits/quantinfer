//! Shared deterministic fixtures for numerical and integration tests.

#![forbid(unsafe_code)]

use quantinfer_core::Tensor;
use quantinfer_model::Qwen3Config;

pub fn tiny_qwen3_config() -> Qwen3Config {
    Qwen3Config {
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
        sliding_window: None,
        max_window_layers: None,
        use_sliding_window: false,
        rope_scaling: None,
        architectures: vec!["Qwen3ForCausalLM".to_owned()],
    }
}

pub fn deterministic_tensor(shape: impl Into<quantinfer_core::Shape>, seed: u32) -> Tensor {
    let shape = shape.into();
    let elements = shape
        .num_elements()
        .expect("deterministic fixture shape must be valid");
    let data = (0..elements)
        .map(|index| {
            let value = ((index as u32).wrapping_mul(1_664_525).wrapping_add(seed)) % 10_000;
            (value as f32 / 10_000.0) - 0.5
        })
        .collect();
    Tensor::from_f32(shape, data).expect("deterministic fixture shape must be valid")
}
