use std::{env, fs};

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_quant::dequantize_q8_0;
use quantinfer_runtime::{
    generate_qwen3_stream, GenerationConfig, KvCache, Qwen3Model, StreamControl,
};
use quantinfer_tokenizer::{
    ChatMessage, GgufBpeTokenizer, Qwen3ChatTemplate, ThinkingMode, Tokenizer,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    prompt: String,
    prompt_tokens: Vec<usize>,
    logits: Vec<f32>,
    eos_token_id: usize,
    generated_tokens: Vec<usize>,
}

#[derive(Deserialize)]
struct Baseline {
    prompt: String,
    prompt_tokens: Vec<usize>,
    eos_token_id: usize,
    max_sequence_length: usize,
    max_new_tokens: usize,
    f32_generated_tokens: Vec<usize>,
    q8_0_generated_tokens: Vec<usize>,
}

#[test]
#[ignore = "requires pinned Qwen3-0.6B F32/Q8_0 conversions and Transformers reference"]
fn matches_qwen3_transformers_reference_and_q8_quality_gates() {
    let reference_path = env::var_os("QUANTINFER_QWEN3_REFERENCE").unwrap();
    let reference: Reference = serde_json::from_slice(&fs::read(reference_path).unwrap()).unwrap();
    let baseline: Baseline = serde_json::from_str(include_str!(
        "../../../tests/golden/qwen3-0.6b-baseline.json"
    ))
    .unwrap();
    assert_eq!(reference.prompt, baseline.prompt);
    assert_eq!(reference.prompt_tokens, baseline.prompt_tokens);
    assert_eq!(reference.eos_token_id, baseline.eos_token_id);
    assert_eq!(reference.generated_tokens, baseline.f32_generated_tokens);

    let f32 = run_variant("QUANTINFER_QWEN3_F32", GgmlType::F32, &baseline);
    let q8 = run_variant("QUANTINFER_QWEN3_Q8", GgmlType::Q8_0, &baseline);

    let official_error = errors(&reference.logits, &f32.logits);
    let quantized_error = errors(&f32.logits, &q8.logits);
    let weight_error = errors(&f32.query_weights, &q8.query_weights);
    let f32_top = top_10(&f32.logits);
    let overlap = top_10(&q8.logits)
        .into_iter()
        .filter(|token| f32_top.contains(token))
        .count();
    println!(
        "f32_vs_transformers={official_error:?} q8_logits={quantized_error:?} q8_top_10_overlap={overlap} q8_query_weights={weight_error:?}"
    );
    println!(
        "f32_generated={:?} q8_generated={:?}",
        f32.generated, q8.generated
    );
    assert_eq!(f32.generated, baseline.f32_generated_tokens);
    assert_eq!(q8.generated, baseline.q8_0_generated_tokens);
    assert_eq!(f32.generated.last(), Some(&baseline.eos_token_id));
    assert_eq!(q8.generated.last(), Some(&baseline.eos_token_id));
    assert!(official_error.max <= 0.001);
    assert_eq!(f32_top, top_10(&reference.logits));
    assert!(quantized_error.max <= 0.9);
    assert!(quantized_error.mse <= 0.05);
    assert!(overlap >= 9);
    assert!(weight_error.max <= 0.003);
    assert!(weight_error.mean <= 0.0002);
}

struct Run {
    logits: Vec<f32>,
    query_weights: Vec<f32>,
    generated: Vec<usize>,
}

fn run_variant(path_var: &str, dtype: GgmlType, baseline: &Baseline) -> Run {
    let path = env::var_os(path_var).unwrap_or_else(|| panic!("set {path_var}"));
    let reader = GgufReader::open(fs::read(path).unwrap()).unwrap();
    let strings = |key: &str| match &reader.metadata()[key] {
        GgufValue::Array(values) => values
            .iter()
            .map(|value| match value {
                GgufValue::String(value) => value.clone(),
                _ => panic!("expected string in {key}"),
            })
            .collect::<Vec<_>>(),
        _ => panic!("expected array in {key}"),
    };
    assert_eq!(
        reader.metadata()["tokenizer.ggml.eos_token_id"],
        GgufValue::U32(baseline.eos_token_id.try_into().unwrap())
    );
    let tokenizer = GgufBpeTokenizer::new(
        strings("tokenizer.ggml.tokens"),
        &strings("tokenizer.ggml.merges"),
        None,
        Some(baseline.eos_token_id),
    )
    .unwrap();
    let chat = Qwen3ChatTemplate
        .render(
            &[ChatMessage::new("user", &baseline.prompt)],
            ThinkingMode::Disabled,
            true,
        )
        .unwrap();
    let prompt = tokenizer.encode(&chat).unwrap();
    assert_eq!(prompt, baseline.prompt_tokens);

    let query_name = "blk.0.attn_q.weight";
    assert_eq!(reader.tensor_info(query_name).unwrap().dtype, dtype);
    let query = match dtype {
        GgmlType::F32 => reader.tensor_f32(query_name).unwrap(),
        GgmlType::Q8_0 => dequantize_q8_0(&reader.tensor_q8_0(query_name).unwrap()).unwrap(),
        _ => unreachable!(),
    };
    let model = Qwen3Model::from_gguf(CpuScalar::new(), &reader).unwrap();
    drop(reader);
    let new_cache = || {
        KvCache::new(
            model.config.num_hidden_layers,
            baseline.max_sequence_length,
            model.config.num_key_value_heads,
            model.config.effective_head_dim(),
        )
        .unwrap()
    };
    let mut cache = new_cache();
    let prefill = model.forward(&prompt, &mut cache).unwrap();
    let vocab = model.config.vocab_size;
    let values = prefill.as_f32().unwrap();
    let logits = values[values.len() - vocab..].to_vec();
    let config = GenerationConfig {
        max_new_tokens: baseline.max_new_tokens,
        eos_token_id: Some(baseline.eos_token_id),
        ..GenerationConfig::default()
    };
    let generate = |config: &GenerationConfig| {
        let mut cache = new_cache();
        let mut streamed = Vec::new();
        let tokens = generate_qwen3_stream(&model, &prompt, &mut cache, config, &mut |token| {
            streamed.push(token);
            Ok(StreamControl::Continue)
        })
        .unwrap();
        assert_eq!(tokens, streamed);
        let eos = tokens.last() == config.eos_token_id.as_ref();
        let cached = prompt.len() + tokens.len() - usize::from(eos);
        for layer in 0..model.config.num_hidden_layers {
            assert_eq!(cache.sequence_length(layer).unwrap(), cached);
        }
        tokens
    };
    let generated = generate(&config);
    assert_eq!(generated, generate(&config));
    assert!(!tokenizer.decode(&generated).unwrap().trim().is_empty());

    Run {
        logits,
        query_weights: query.as_f32().unwrap().to_vec(),
        generated,
    }
}

fn top_10(values: &[f32]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(10);
    indices
}

#[derive(Debug)]
struct Errors {
    max: f32,
    mean: f64,
    mse: f64,
}

fn errors(reference: &[f32], actual: &[f32]) -> Errors {
    assert_eq!(reference.len(), actual.len());
    let mut max = 0.0_f32;
    let mut sum = 0.0;
    let mut squared_sum = 0.0;
    for (&expected, &actual) in reference.iter().zip(actual) {
        let error = (actual - expected).abs();
        assert!(error.is_finite());
        max = max.max(error);
        sum += f64::from(error);
        squared_sum += f64::from(error).powi(2);
    }
    let count = reference.len() as f64;
    Errors {
        max,
        mean: sum / count,
        mse: squared_sum / count,
    }
}
