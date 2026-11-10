use std::{env, fs};

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_format::{GgufReader, GgufValue};
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
#[ignore = "requires the converted SmolLM2-135M F32 artifact"]
fn matches_smollm2_transformers_reference() {
    const MAX_LOGIT_ERROR: f32 = 0.001;
    let golden: Golden =
        serde_json::from_str(include_str!("../../../tests/golden/smollm2-135m-f32.json")).unwrap();
    let path = env::var_os("QUANTINFER_SMOLLM2_GGUF").expect("set QUANTINFER_SMOLLM2_GGUF");
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
    assert_eq!(&top[..golden.top_10.len()], golden.top_10);
    let max_error = golden
        .logits
        .iter()
        .map(|&(index, expected)| {
            assert!(prefill[index].is_finite());
            (prefill[index] - expected).abs()
        })
        .fold(0.0, f32::max);
    println!("SmolLM2 F32 sampled logit max error = {max_error:.8}");
    assert!(
        max_error <= MAX_LOGIT_ERROR,
        "sampled logit error {max_error} exceeds {MAX_LOGIT_ERROR}"
    );

    let mut generated = Vec::new();
    for step in 0..golden.generated_tokens.len() {
        let row = last_row(logits.as_f32().unwrap(), vocab);
        let (next, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .unwrap();
        generated.push(next);
        if next == golden.eos_token_id || step + 1 == golden.generated_tokens.len() {
            break;
        }
        logits = model.forward(&[next], &mut cache).unwrap();
    }
    assert_eq!(generated, golden.generated_tokens);
    assert_eq!(tokenizer.decode(&generated).unwrap(), golden.generated_text);
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
