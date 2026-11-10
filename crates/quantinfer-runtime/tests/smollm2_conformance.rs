use std::{env, fs};

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_format::{GgmlType, GgufReader, GgufValue};
use quantinfer_runtime::{KvCache, LlamaModel};
use quantinfer_tokenizer::{GgufBpeTokenizer, Tokenizer};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    prompt: String,
    prompt_tokens: Vec<usize>,
    eos_token_id: usize,
    top_10: Vec<usize>,
    logits: Vec<(usize, f32)>,
    generated_tokens: Vec<usize>,
    generated_text: String,
}

#[test]
#[ignore = "requires a pinned SmolLM2-135M GGUF artifact"]
fn matches_smollm2_transformers_reference() {
    let golden: Golden =
        serde_json::from_str(include_str!("../../../tests/golden/smollm2-135m-f32.json")).unwrap();
    let path = env::var_os("QUANTINFER_SMOLLM2_GGUF").expect("set QUANTINFER_SMOLLM2_GGUF");
    let variant = env::var("QUANTINFER_SMOLLM2_VARIANT").unwrap_or_else(|_| "float".to_owned());
    let (quant_dtype, max_logit_error, min_overlap, expected): (_, _, _, &[usize]) =
        match variant.as_str() {
            "float" => (None, 0.001, 10, &golden.generated_tokens),
            "q8_0" => (Some(GgmlType::Q8_0), 0.15, 9, &golden.generated_tokens),
            "q4_0" => (
                Some(GgmlType::Q4_0),
                3.8,
                6,
                &[260, 3432, 1789, 30, 657, 314, 253, 8196],
            ),
            _ => panic!("unsupported SmolLM2 conformance variant {variant}"),
        };
    let reader = GgufReader::open(fs::read(path).unwrap()).unwrap();
    let dtype = reader.tensor_info("blk.0.attn_q.weight").unwrap().dtype;
    if let Some(expected_dtype) = quant_dtype {
        assert_eq!(dtype, expected_dtype);
    } else {
        assert!(matches!(
            dtype,
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
        ));
    }
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
        GgufValue::U32(golden.eos_token_id.try_into().unwrap())
    );
    let tokenizer = GgufBpeTokenizer::new(
        strings("tokenizer.ggml.tokens"),
        &strings("tokenizer.ggml.merges"),
        None,
        Some(golden.eos_token_id),
    )
    .unwrap();
    let prompt = tokenizer.encode(&golden.prompt).unwrap();
    assert_eq!(prompt, golden.prompt_tokens);

    let model = LlamaModel::from_gguf(CpuScalar::new(), &reader).unwrap();
    drop(reader);
    let mut cache = KvCache::new(
        model.config.num_hidden_layers,
        64,
        model.config.num_key_value_heads(),
        model.config.effective_head_dim(),
    )
    .unwrap();
    let mut logits = model.forward(&prompt, &mut cache).unwrap();
    let vocab = model.config.vocab_size;
    let prefill = last_row(logits.as_f32().unwrap(), vocab);
    let mut top = (0..vocab).collect::<Vec<_>>();
    top.sort_by(|&left, &right| prefill[right].total_cmp(&prefill[left]));
    let top = &top[..golden.top_10.len()];
    let overlap = top
        .iter()
        .filter(|token| golden.top_10.contains(token))
        .count();
    let max_error = golden
        .logits
        .iter()
        .map(|&(index, expected)| {
            assert!(prefill[index].is_finite());
            (prefill[index] - expected).abs()
        })
        .fold(0.0, f32::max);
    println!("SmolLM2 {variant} sampled_logit_max_error={max_error:.8} top_10_overlap={overlap}");
    assert!(
        max_error <= max_logit_error,
        "sampled logit error {max_error} exceeds {max_logit_error}"
    );
    assert!(
        overlap >= min_overlap,
        "top-10 overlap {overlap} is below {min_overlap}"
    );
    if quant_dtype.is_none() {
        assert_eq!(top, golden.top_10);
    }

    let mut generated = Vec::new();
    for step in 0..expected.len() {
        let row = last_row(logits.as_f32().unwrap(), vocab);
        let (next, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .unwrap();
        generated.push(next);
        if next == golden.eos_token_id || step + 1 == expected.len() {
            break;
        }
        logits = model.forward(&[next], &mut cache).unwrap();
    }
    println!("SmolLM2 {variant} generated_tokens={generated:?}");
    assert_eq!(generated, expected);
    if quant_dtype.is_none() {
        assert_eq!(tokenizer.decode(&generated).unwrap(), golden.generated_text);
    }
    for layer in 0..model.config.num_hidden_layers {
        assert_eq!(
            cache.sequence_length(layer).unwrap(),
            prompt.len() + generated.len() - 1
        );
    }
}

fn last_row(values: &[f32], width: usize) -> &[f32] {
    &values[values.len() - width..]
}
