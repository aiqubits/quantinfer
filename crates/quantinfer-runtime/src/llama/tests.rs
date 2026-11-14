use super::{LlamaLayerWeights, LlamaModel};
use crate::{generate_stream, GenerationConfig, KvCache, StreamControl};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use quantinfer_model::LlamaConfig;

fn filled(shape: Vec<usize>, value: f32) -> Tensor {
    Tensor::from_f32(shape.clone(), vec![value; shape.iter().product()]).unwrap()
}

fn config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: Some(1),
        head_dim: Some(4),
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 32,
        tie_word_embeddings: true,
        architectures: vec!["LlamaForCausalLM".to_owned()],
    }
}

#[test]
fn runs_dense_llama_prefill_and_decode_with_gqa() {
    let config = config();
    let model = LlamaModel::new(
        config.clone(),
        CpuScalar::new(),
        filled(vec![config.vocab_size, config.hidden_size], 0.01),
        vec![LlamaLayerWeights {
            input_layernorm: filled(vec![config.hidden_size], 1.0),
            q_proj: super::LinearWeight::F32(filled(vec![8, 8], 0.01)),
            k_proj: super::LinearWeight::F32(filled(vec![8, 4], 0.01)),
            v_proj: super::LinearWeight::F32(filled(vec![8, 4], 0.01)),
            o_proj: super::LinearWeight::F32(filled(vec![8, 8], 0.01)),
            post_attention_layernorm: filled(vec![config.hidden_size], 1.0),
            gate_proj: super::LinearWeight::F32(filled(vec![8, 16], 0.01)),
            up_proj: super::LinearWeight::F32(filled(vec![8, 16], 0.01)),
            down_proj: super::LinearWeight::F32(filled(vec![16, 8], 0.01)),
        }],
        filled(vec![config.hidden_size], 1.0),
        filled(vec![config.hidden_size, config.vocab_size], 0.01),
    )
    .unwrap();
    let mut cache = KvCache::new(1, 32, 1, 4).unwrap();
    let prefill = model.forward(&[1, 2], &mut cache).unwrap();
    assert_eq!(prefill.shape().dims(), &[2, config.vocab_size]);
    let decode = model.forward(&[3], &mut cache).unwrap();
    assert_eq!(decode.shape().dims(), &[1, config.vocab_size]);
    assert_eq!(cache.sequence_length(0).unwrap(), 3);

    cache.clear();
    let mut emitted = Vec::new();
    let generated = generate_stream(
        &model,
        &[1, 2],
        &mut cache,
        &GenerationConfig {
            max_new_tokens: 4,
            eos_token_id: Some(0),
            ..GenerationConfig::default()
        },
        &mut |token| {
            emitted.push(token);
            Ok(StreamControl::Continue)
        },
    )
    .unwrap();
    assert_eq!(emitted, [0]);
    assert_eq!(generated, emitted);
    assert_eq!(cache.sequence_length(0).unwrap(), 2);
}
