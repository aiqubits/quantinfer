//! Fixed-model scalar reference and publication checks shared by quality matrices.
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_format::{
    is_linear_projection_storage_name,
    qif::{FileReader, PackageReader},
    sha256_hex, GgmlType, PackedTensorReader,
};
use quantinfer_quant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use quantinfer_runtime::{
    generate_stream, DecoderModel, GenerationConfig, KvCache, LlamaModel, Qwen3Model, StreamControl,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
#[derive(Deserialize)]
pub(super) struct Fixture {
    pub(super) prompt_tokens: Vec<usize>,
    pub(super) eos_token_id: usize,
    pub(super) logits: Option<Vec<(usize, f32)>>,
}

struct Geometry {
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
}
pub(super) struct Run {
    pub(super) logits: Vec<f32>,
    pub(super) query: Vec<f32>,
    pub(super) inventory: BTreeMap<String, (Vec<usize>, GgmlType, String)>,
    pub(super) generated: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(super) enum Container {
    Gguf,
    SingleNone,
    PackageZstd,
}

pub(super) fn run(
    path: &Path,
    kind: &str,
    expected: &dyn Fn(&str) -> GgmlType,
    fixture: &Fixture,
    container: Container,
) -> Run {
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
                let dtype = expected(&tensor.name);
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
    let query = match reader.tensor_info(name).unwrap().dtype {
        GgmlType::F32 => reader.tensor_f32(name).unwrap(),
        GgmlType::Q8_0 => dequantize_q8_0(&reader.tensor_q8_0(name).unwrap()).unwrap(),
        GgmlType::Q4_0 => dequantize_q4_0(&reader.tensor_q4_0(name).unwrap()).unwrap(),
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
pub(super) struct Errors {
    pub(super) max: f64,
    pub(super) mean: f64,
    pub(super) mse: f64,
    pub(super) relative_l2: f64,
}
pub(super) fn errors(reference: &[f32], actual: &[f32]) -> Errors {
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
pub(super) fn top(values: &[f32]) -> Vec<usize> {
    let mut ids = (0..values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    ids.truncate(10);
    ids
}

pub(super) fn probability_kl(reference: &[f32], actual: &[f32]) -> f64 {
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
