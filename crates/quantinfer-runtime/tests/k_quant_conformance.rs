//! Fixed-model native-layout K-quant quality and QIF publication matrix.
//! Existing model-specific tests independently anchor the F32 baseline to
//! Transformers; this test adds full-vocabulary comparisons and wire parity.
use std::{collections::BTreeMap, env, fs, path::Path};

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_format::{
    is_linear_projection_storage_name,
    qif::{FileReader, PackageReader},
    sha256_hex, GgmlType, PackedTensorReader,
};
use quantinfer_quant::{dequantize_q4_k, dequantize_q5_k, dequantize_q6_k};
use quantinfer_runtime::{
    generate_stream, DecoderModel, GenerationConfig, KvCache, LlamaModel, Qwen3Model, StreamControl,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[path = "support/qif_publication.rs"]
mod qif_publication;

#[derive(Deserialize)]
struct Fixture {
    prompt_tokens: Vec<usize>,
    eos_token_id: usize,
    logits: Option<Vec<(usize, f32)>>,
}

struct Geometry {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
}
struct Run {
    logits: Vec<f32>,
    query: Vec<f32>,
    inventory: BTreeMap<String, (Vec<usize>, GgmlType, String)>,
    generated: Vec<usize>,
}

#[derive(Clone, Copy)]
enum Container {
    Gguf,
    SingleNone,
    PackageZstd,
}

fn run(path: &Path, kind: &str, dtype: GgmlType, fixture: &Fixture, container: Container) -> Run {
    let reader = PackedTensorReader::open_path(path).unwrap();
    assert!(!reader.uses_ggml_output_rows().unwrap());
    match container {
        Container::Gguf => assert!(!reader.is_file_backed()),
        Container::SingleNone => {
            assert!(reader.is_file_backed());
            let controls = FileReader::open(path).unwrap();
            assert!(controls.tensors().iter().all(|tensor| tensor.codec == 0));
            reader.verify_qif().unwrap();
        }
        Container::PackageZstd => {
            assert!(reader.is_file_backed());
            let controls = PackageReader::open(path).unwrap();
            assert!(controls.tensors().iter().all(|tensor| tensor.codec == 1));
            reader.verify_qif().unwrap();
        }
    }
    let inventory = reader
        .tensors()
        .iter()
        .map(|tensor| {
            if is_linear_projection_storage_name(&tensor.name) {
                assert_eq!(tensor.dtype, dtype, "{}", tensor.name);
            }
            (
                tensor.name.clone(),
                (
                    tensor.shape.clone(),
                    tensor.dtype,
                    sha256_hex(reader.tensor_data(&tensor.name).unwrap().as_ref()),
                ),
            )
        })
        .collect();
    let name = "blk.0.attn_q.weight";
    assert_eq!(reader.tensor_info(name).unwrap().dtype, dtype);
    let query = match dtype {
        GgmlType::F32 => reader.tensor_f32(name).unwrap(),
        GgmlType::Q4_K => dequantize_q4_k(&reader.tensor_q4_k(name).unwrap()).unwrap(),
        GgmlType::Q5_K => dequantize_q5_k(&reader.tensor_q5_k(name).unwrap()).unwrap(),
        GgmlType::Q6_K => dequantize_q6_k(&reader.tensor_q6_k(name).unwrap()).unwrap(),
        _ => unreachable!(),
    };
    let (model, geometry): (Box<dyn DecoderModel>, Geometry) = match kind {
        "smollm2" => {
            let model = LlamaModel::from_gguf(CpuScalar::new(), &reader).unwrap();
            let c = &model.config;
            let geometry = Geometry {
                layers: c.num_hidden_layers,
                kv_heads: c.num_key_value_heads(),
                head_dim: c.effective_head_dim(),
                vocab: c.vocab_size,
            };
            (Box::new(model), geometry)
        }
        "qwen3" => {
            let model = Qwen3Model::from_gguf(CpuScalar::new(), &reader).unwrap();
            let c = &model.config;
            let geometry = Geometry {
                layers: c.num_hidden_layers,
                kv_heads: c.num_key_value_heads,
                head_dim: c.effective_head_dim(),
                vocab: c.vocab_size,
            };
            (Box::new(model), geometry)
        }
        _ => panic!("unsupported fixed model {kind}"),
    };
    assert_eq!(reader.cached_tensor_bytes(), 0);
    drop(reader);
    let cache = || KvCache::new(geometry.layers, 64, geometry.kv_heads, geometry.head_dim).unwrap();
    let prefill = model.forward(&fixture.prompt_tokens, &mut cache()).unwrap();
    let values = prefill.as_f32().unwrap();
    assert_eq!(
        prefill.shape().dims(),
        [fixture.prompt_tokens.len(), geometry.vocab]
    );
    assert!(values.iter().all(|value| value.is_finite()));
    let logits = values[values.len() - geometry.vocab..].to_vec();
    let generate = || {
        let mut cache = cache();
        let mut streamed = Vec::new();
        let config = GenerationConfig {
            max_new_tokens: 8,
            eos_token_id: Some(fixture.eos_token_id),
            ..Default::default()
        };
        let generated = generate_stream(
            &*model,
            &fixture.prompt_tokens,
            &mut cache,
            &config,
            &mut |token| {
                streamed.push(token);
                Ok(StreamControl::Continue)
            },
        )
        .unwrap();
        assert_eq!(generated, streamed);
        assert!(!generated.is_empty() && generated.len() <= 8);
        assert!(generated.iter().all(|&token| token < geometry.vocab));
        let expected = fixture.prompt_tokens.len() + generated.len()
            - usize::from(generated.last() == Some(&fixture.eos_token_id));
        for layer in 0..geometry.layers {
            assert_eq!(cache.sequence_length(layer).unwrap(), expected);
        }
        generated
    };
    let generated = generate();
    assert_eq!(generated, generate());
    Run {
        logits,
        query: query.as_f32().unwrap().to_vec(),
        inventory,
        generated,
    }
}

#[derive(Debug, Serialize)]
struct Errors {
    max: f64,
    mean: f64,
    mse: f64,
    relative_l2: f64,
}
fn errors(reference: &[f32], actual: &[f32]) -> Errors {
    assert_eq!(reference.len(), actual.len());
    assert!(!reference.is_empty());
    let (mut max, mut sum, mut square, mut norm) = (0.0f64, 0.0, 0.0, 0.0);
    for (&left, &right) in reference.iter().zip(actual) {
        assert!(left.is_finite() && right.is_finite());
        let left = f64::from(left);
        let difference = (left - f64::from(right)).abs();
        max = max.max(difference);
        sum += difference;
        square += difference * difference;
        norm += left * left;
    }
    assert!(norm > 0.0);
    Errors {
        max,
        mean: sum / reference.len() as f64,
        mse: square / reference.len() as f64,
        relative_l2: (square / norm).sqrt(),
    }
}
fn top(values: &[f32]) -> Vec<usize> {
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    ids.truncate(10);
    ids
}

fn probability_kl(reference: &[f32], actual: &[f32]) -> f64 {
    assert_eq!(reference.len(), actual.len());
    assert!(!reference.is_empty());
    let log_probabilities = |values: &[f32]| {
        assert!(values.iter().all(|value| value.is_finite()));
        let max = values
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = values
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
        values
            .iter()
            .map(|&v| (f64::from(v) - max) - log_sum)
            .collect::<Vec<_>>()
    };
    let p = log_probabilities(reference);
    let q = log_probabilities(actual);
    let divergence = p
        .iter()
        .zip(q)
        .map(|(&p, q)| p.exp() * (p - q))
        .sum::<f64>();
    assert!(divergence.is_finite() && divergence >= -1.0e-12);
    divergence.max(0.0)
}

#[test]
fn probability_error_ignores_constant_logit_shifts() {
    let baseline = [1.0, 0.0, -1.0];
    let shifted = [21.0, 20.0, 19.0];
    assert!(errors(&baseline, &shifted).max > 10.0);
    assert!(probability_kl(&baseline, &shifted) < 1.0e-12);
    assert!(probability_kl(&[10.0, -10.0], &[-10.0, 10.0]) > 19.0);
    assert!(probability_kl(&[f32::MAX; 2], &[f32::MAX; 2]) < 1.0e-12);
}

#[test]
#[ignore = "requires verified SmolLM2/Qwen3 native-layout conversions"]
fn validates_native_k_quant_quality_and_qif_parity() {
    let kind = env::var("QUANTINFER_K_QUANT_MODEL").expect("set smollm2 or qwen3");
    let root = env::var_os("QUANTINFER_K_QUANT_DIR").expect("set verified artifact directory");
    let root = Path::new(&root);
    let fixture: Fixture = serde_json::from_str(match kind.as_str() {
        "smollm2" => include_str!("../../../tests/golden/smollm2-135m-f32.json"),
        "qwen3" => include_str!("../../../tests/golden/qwen3-0.6b-baseline.json"),
        _ => panic!("unsupported fixed model {kind}"),
    })
    .unwrap();
    let baseline = run(
        &root.join(format!("{kind}-f32.gguf")),
        &kind,
        GgmlType::F32,
        &fixture,
        Container::Gguf,
    );
    let baseline_top = top(&baseline.logits);
    let mut records = Vec::new();
    let mut violations = Vec::new();
    for (variant, dtype) in [
        ("q4_k", GgmlType::Q4_K),
        ("q5_k", GgmlType::Q5_K),
        ("q6_k", GgmlType::Q6_K),
    ] {
        let quantized = run(
            &root.join(format!("{kind}-{variant}.gguf")),
            &kind,
            dtype,
            &fixture,
            Container::Gguf,
        );
        let logits = errors(&baseline.logits, &quantized.logits);
        let weights = errors(&baseline.query, &quantized.query);
        let kl = probability_kl(&baseline.logits, &quantized.logits);
        let overlap = top(&quantized.logits)
            .iter()
            .filter(|id| baseline_top.contains(id))
            .count();
        // Bound next-token distribution divergence for this fixed prompt.
        // The separate legacy model tests retain their existing raw-logit
        // budgets. Raw logits also remain in this report: different layouts
        // and constant shifts do not imply the same probability error.
        if kl > 0.5 {
            violations.push(format!(
                "{kind} {variant} next-token KL {kl} exceeds 0.5 nats"
            ));
        }
        if overlap < 6 {
            violations.push(format!("{kind} {variant} top-10 overlap {overlap}"));
        }
        if kind == "qwen3" {
            if logits.max > 13.0 || logits.mse > 9.0 {
                violations.push(format!("{variant} logits {logits:?}"));
            }
            if weights.max > 0.032 || weights.mean > 0.0025 {
                violations.push(format!("{variant} weights {weights:?}"));
            }
        }
        let sampled = fixture.logits.as_ref().map(|values| {
            values
                .iter()
                .map(|&(id, value)| (f64::from(quantized.logits[id]) - f64::from(value)).abs())
                .fold(0.0, f64::max)
        });
        if kind == "smollm2" && weights.relative_l2 > 0.25 {
            violations.push(format!("{variant} weights {weights:?}"));
        }
        let mut record = json!({"model":kind,"variant":variant,"logits":logits,"sampled_transformers_max_error":sampled,"probability_kl_nats":kl,"query_weights":weights,"top_10_overlap":overlap,"generated":quantized.generated});
        if env::var_os("QUANTINFER_K_QUANT_QIF").is_some()
            || env::var_os("QUANTINFER_K_QUANT_CONVERTER").is_some()
        {
            let artifacts = qif_publication::Artifacts::open_or_publish(root, &kind, variant);
            for (path, profile, container) in [
                (&artifacts.single, "single_none", Container::SingleNone),
                (&artifacts.package, "package_zstd", Container::PackageZstd),
            ] {
                let qif = run(path, &kind, dtype, &fixture, container);
                assert_eq!(qif.inventory, quantized.inventory);
                assert_eq!(qif.query, quantized.query);
                assert_eq!(qif.logits, quantized.logits);
                assert_eq!(qif.generated, quantized.generated);
                println!("k_quant_qif_parity model={kind} variant={variant} profile={profile}");
            }
            record["qif_artifact_sha256"] = serde_json::to_value(artifacts.hashes()).unwrap();
            artifacts.cleanup();
        }
        println!("k_quant_quality={record}");
        records.push(record);
    }
    if let Some(path) = env::var_os("QUANTINFER_K_QUANT_REPORT") {
        fs::write(path,serde_json::to_vec_pretty(&json!({"source_commit":env::var("GITHUB_SHA").ok(),"model":kind,"records":records,"passed":violations.is_empty(),"violations":violations})).unwrap()).unwrap();
    }
    assert!(
        violations.is_empty(),
        "quality budget failures: {violations:?}"
    );
}
