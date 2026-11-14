use super::{LinearWeight, Qwen3LayerWeights, Qwen3Model};
use crate::{generate_qwen3_stream, GenerationConfig, KvCache, StreamControl};
use quantinfer_backend_cpu::{CpuRuntimeConfig, CpuScalar};
use quantinfer_core::{DType, Tensor};
use quantinfer_format::{GgufReader, GgufValue, GgufWriter};
use quantinfer_model::Qwen3Config;
use quantinfer_quant::{quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0};

fn filled(shape: Vec<usize>, value: f32) -> Tensor {
    let count = shape.iter().product();
    Tensor::from_f32(shape, vec![value; count]).unwrap()
}

fn linear(shape: Vec<usize>, value: f32) -> LinearWeight {
    LinearWeight::F32(filled(shape, value))
}

fn quantized_linear(shape: Vec<usize>, value: f32) -> LinearWeight {
    LinearWeight::Q8_0(quantize_q8_0(&filled(shape, value)).unwrap())
}

fn config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(4),
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        max_position_embeddings: 32,
        tie_word_embeddings: true,
        sliding_window: None,
        max_window_layers: None,
        use_sliding_window: false,
        rope_scaling: None,
        architectures: vec!["Qwen3ForCausalLM".to_owned()],
    }
}

fn layer() -> Qwen3LayerWeights {
    Qwen3LayerWeights {
        input_layernorm: filled(vec![8], 1.0),
        q_proj: linear(vec![8, 8], 0.01),
        q_norm: filled(vec![4], 1.0),
        k_proj: linear(vec![8, 4], 0.01),
        k_norm: filled(vec![4], 1.0),
        v_proj: linear(vec![8, 4], 0.01),
        o_proj: linear(vec![8, 8], 0.01),
        post_attention_layernorm: filled(vec![8], 1.0),
        gate_proj: linear(vec![8, 16], 0.01),
        up_proj: linear(vec![8, 16], 0.01),
        down_proj: linear(vec![16, 8], 0.01),
    }
}

fn quantized_layer() -> Qwen3LayerWeights {
    Qwen3LayerWeights {
        input_layernorm: filled(vec![8], 1.0),
        q_proj: quantized_linear(vec![8, 8], 0.01),
        q_norm: filled(vec![4], 1.0),
        k_proj: quantized_linear(vec![8, 4], 0.01),
        k_norm: filled(vec![4], 1.0),
        v_proj: quantized_linear(vec![8, 4], 0.01),
        o_proj: quantized_linear(vec![8, 8], 0.01),
        post_attention_layernorm: filled(vec![8], 1.0),
        gate_proj: quantized_linear(vec![8, 16], 0.01),
        up_proj: quantized_linear(vec![8, 16], 0.01),
        down_proj: quantized_linear(vec![16, 8], 0.01),
    }
}

#[test]
fn runs_qwen3_prefill_and_decode() {
    let config = config();
    let model = Qwen3Model::new(
        config.clone(),
        CpuScalar::new(),
        filled(vec![config.vocab_size, config.hidden_size], 0.01),
        vec![layer()],
        filled(vec![config.hidden_size], 1.0),
        filled(vec![config.hidden_size, config.vocab_size], 0.01),
    )
    .unwrap();
    let mut cache = KvCache::new(
        config.num_hidden_layers,
        config.max_position_embeddings,
        config.num_key_value_heads,
        config.effective_head_dim(),
    )
    .unwrap();
    let prefill = model.forward(&[1, 2], &mut cache).unwrap();
    assert_eq!(prefill.shape().dims(), &[2, config.vocab_size]);
    assert_eq!(cache.sequence_length(0).unwrap(), 2);
    let decode = model.forward(&[3], &mut cache).unwrap();
    assert_eq!(decode.shape().dims(), &[1, config.vocab_size]);
    assert_eq!(cache.sequence_length(0).unwrap(), 3);
}

#[test]
fn attention_workspace_reuse_matches_fresh_scores() {
    let mut config = config();
    config.num_hidden_layers = 2;
    let model = |workspace_reuse| {
        Qwen3Model::new(
            config.clone(),
            CpuScalar::with_config(CpuRuntimeConfig {
                workspace_reuse,
                ..CpuScalar::new().config()
            }),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![layer(), layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap()
    };
    let reused = model(true);
    let fresh = model(false);
    let mut reused_cache = KvCache::new(2, config.max_position_embeddings, 1, 4).unwrap();
    let mut fresh_cache = reused_cache.clone();
    for prompt in [&[1, 2][..], &[3][..], &[4][..]] {
        assert_eq!(
            reused.forward(prompt, &mut reused_cache).unwrap(),
            fresh.forward(prompt, &mut fresh_cache).unwrap()
        );
    }
    assert_eq!(fresh_cache.attention_scores.capacity(), 0);
    assert!(reused_cache.attention_scores.capacity() >= 4);
    let scratch = reused_cache.attention_scores.as_ptr();
    reused_cache.clear();
    reused.forward(&[1, 2], &mut reused_cache).unwrap();
    assert_eq!(reused_cache.attention_scores.as_ptr(), scratch);
}

#[test]
fn streams_tokens_and_honors_sink_stop() {
    let config = config();
    let model = Qwen3Model::new(
        config.clone(),
        CpuScalar::new(),
        filled(vec![config.vocab_size, config.hidden_size], 0.01),
        vec![layer()],
        filled(vec![config.hidden_size], 1.0),
        filled(vec![config.hidden_size, config.vocab_size], 0.01),
    )
    .unwrap();
    let mut cache = KvCache::new(1, config.max_position_embeddings, 1, 4).unwrap();
    let mut streamed = Vec::new();
    let generated = generate_qwen3_stream(
        &model,
        &[1, 2],
        &mut cache,
        &GenerationConfig {
            max_new_tokens: 4,
            ..GenerationConfig::default()
        },
        &mut |token| {
            streamed.push(token);
            Ok(StreamControl::Stop)
        },
    )
    .unwrap();
    assert_eq!(generated, streamed);
    assert_eq!(streamed.len(), 1);
}

#[test]
fn runs_qwen3_with_q8_linear_weights() {
    let config = config();
    let model = Qwen3Model::new(
        config.clone(),
        CpuScalar::new(),
        filled(vec![config.vocab_size, config.hidden_size], 0.01),
        vec![quantized_layer()],
        filled(vec![config.hidden_size], 1.0),
        filled(vec![config.hidden_size, config.vocab_size], 0.01),
    )
    .unwrap();
    let mut cache = KvCache::new(
        config.num_hidden_layers,
        config.max_position_embeddings,
        config.num_key_value_heads,
        config.effective_head_dim(),
    )
    .unwrap();
    let logits = model.forward(&[1], &mut cache).unwrap();
    assert_eq!(logits.shape().dims(), &[1, config.vocab_size]);
}

#[test]
fn loads_qwen3_from_gguf_with_q8_projection() {
    let config = config();
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("qwen3".to_owned()),
    );
    writer.insert_metadata("qwen3.vocab_size", GgufValue::U32(16));
    writer.insert_metadata("qwen3.context_length", GgufValue::U32(32));
    writer.insert_metadata("qwen3.embedding_length", GgufValue::U32(8));
    writer.insert_metadata("qwen3.feed_forward_length", GgufValue::U32(16));
    writer.insert_metadata("qwen3.block_count", GgufValue::U32(1));
    writer.insert_metadata("qwen3.attention.head_count", GgufValue::U32(2));
    writer.insert_metadata("qwen3.attention.head_count_kv", GgufValue::U32(1));
    writer.insert_metadata("qwen3.attention.head_dim", GgufValue::U32(4));
    writer.insert_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        GgufValue::F32(1e-6),
    );
    writer.insert_metadata("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0));
    writer.insert_metadata("qwen3.tie_word_embeddings", GgufValue::Bool(true));
    writer.insert_metadata(
        "qwen3.architectures",
        GgufValue::Array(vec![GgufValue::String("Qwen3ForCausalLM".to_owned())]),
    );
    writer
        .add_f32_tensor("token_embd.weight", vec![16, 8], &vec![0.01; 128])
        .unwrap();
    writer
        .add_f32_tensor("output_norm.weight", vec![8], &[1.0; 8])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.attn_norm.weight", vec![8], &[1.0; 8])
        .unwrap();
    writer.add_q8_0_tensor(
        "blk.0.attn_q.weight",
        &quantize_q8_0(&filled(vec![8, 8], 0.01)).unwrap(),
    );
    writer
        .add_f32_tensor("blk.0.attn_q_norm.weight", vec![4], &[1.0; 4])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.attn_k.weight", vec![8, 4], &[0.01; 32])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.attn_k_norm.weight", vec![4], &[1.0; 4])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.attn_v.weight", vec![8, 4], &[0.01; 32])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.attn_output.weight", vec![8, 8], &vec![0.01; 64])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.ffn_norm.weight", vec![8], &[1.0; 8])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.ffn_gate.weight", vec![8, 16], &vec![0.01; 128])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.ffn_up.weight", vec![8, 16], &vec![0.01; 128])
        .unwrap();
    writer
        .add_f32_tensor("blk.0.ffn_down.weight", vec![16, 8], &vec![0.01; 128])
        .unwrap();

    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let model = Qwen3Model::from_gguf(CpuScalar::new(), &reader).unwrap();
    assert_eq!(model.config, config);
    assert_eq!(model.layers[0].q_proj.dtype(), DType::Q8_0);
    assert_eq!(model.lm_head.shape().dims(), &[8, 16]);
    let mut cache = KvCache::new(1, 32, 1, 4).unwrap();
    let logits = model.forward(&[1, 2], &mut cache).unwrap();
    assert_eq!(logits.shape().dims(), &[2, 16]);
}

#[test]
fn loads_q4_linear_weight_from_gguf() {
    let mut writer = GgufWriter::new();
    writer.add_q4_0_tensor(
        "blk.0.attn_q.weight",
        &quantize_q4_0(&filled(vec![32, 1], 0.25)).unwrap(),
    );
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let linear = crate::linear::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
    assert_eq!(linear.dtype(), DType::Q4_0);
}

#[test]
fn loads_q4_k_linear_weight_from_gguf() {
    let mut writer = GgufWriter::new();
    writer.add_q4_k_tensor(
        "blk.0.attn_q.weight",
        &quantize_q4_k(&filled(vec![256, 1], 0.25)).unwrap(),
    );
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let linear = crate::linear::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
    assert_eq!(linear.dtype(), DType::Q4_K);
}

#[test]
fn loads_q5_k_linear_weight_from_gguf() {
    let mut writer = GgufWriter::new();
    writer.add_q5_k_tensor(
        "blk.0.attn_q.weight",
        &quantize_q5_k(&filled(vec![256, 1], 0.25)).unwrap(),
    );
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let linear = crate::linear::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
    assert_eq!(linear.dtype(), DType::Q5_K);
}

#[test]
fn loads_q6_k_linear_weight_from_gguf() {
    let mut writer = GgufWriter::new();
    writer.add_q6_k_tensor(
        "blk.0.attn_q.weight",
        &quantize_q6_k(&filled(vec![256, 1], 0.25)).unwrap(),
    );
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let linear = crate::linear::load_linear(&reader, "blk.0.attn_q.weight").unwrap();
    assert_eq!(linear.dtype(), DType::Q6_K);
}
