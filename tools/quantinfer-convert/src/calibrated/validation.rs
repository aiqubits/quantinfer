//! Holdout diagnostics and exact publish/reload parity for offline quantization.

use quantinfer::{
    cpu::CpuModel,
    runtime::{generate_stream, DecoderModel, GenerationConfig, StreamControl},
};
use quantinfer_core::Tensor;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(super) struct Snapshot {
    pub tokens: Vec<usize>,
    pub logits: Vec<f32>,
    pub forward_sha256: String,
    pub generated: Vec<usize>,
}

pub(super) fn capture(
    model: &CpuModel,
    prompts: &[Vec<usize>],
    context: usize,
    eos: Option<usize>,
) -> Result<Vec<Snapshot>, String> {
    let mut result = Vec::new();
    for tokens in prompts {
        if tokens.len() > context {
            return Err("validation prompt exceeds the context capacity".into());
        }
        let mut cache = model.new_cache(context).map_err(|e| e.to_string())?;
        let logits = model
            .forward(tokens, &mut cache)
            .map_err(|e| e.to_string())?;
        let values = logits.as_f32().map_err(|e| e.to_string())?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("validation produced non-finite logits".into());
        }
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        let generated = if tokens.len() < context {
            let config = GenerationConfig {
                max_new_tokens: 8.min(context - tokens.len()),
                eos_token_id: eos,
                ..Default::default()
            };
            drop(cache);
            cache = model.new_cache(context).map_err(|e| e.to_string())?;
            generate_stream(model, tokens, &mut cache, &config, &mut |_| {
                Ok(StreamControl::Continue)
            })
            .map_err(|e| e.to_string())?
        } else {
            Vec::new()
        };
        let cached = tokens.len() + generated.len()
            - usize::from(!generated.is_empty() && generated.last().copied() == eos);
        let layers = match model {
            CpuModel::Qwen3(model) => model.layers.len(),
            CpuModel::Llama(model) => model.layers.len(),
        };
        for layer in 0..layers {
            if cache.sequence_length(layer).map_err(|e| e.to_string())? != cached {
                return Err("validation KV state is inconsistent".into());
            }
        }
        result.push(Snapshot {
            tokens: tokens.clone(),
            logits: values[values.len() - model.vocab_size()..].to_vec(),
            forward_sha256: hex(hash),
            generated,
        });
    }
    Ok(result)
}

fn log_sum(values: &[f32]) -> f64 {
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    peak + values
        .iter()
        .map(|&v| (f64::from(v) - peak).exp())
        .sum::<f64>()
        .ln()
}

pub(super) fn compare(
    reference: &[Snapshot],
    actual: &[Snapshot],
    maximum_kl: f64,
    minimum_top: usize,
) -> Result<Vec<Value>, String> {
    if reference.len() != actual.len() {
        return Err("validation sample count changed".into());
    }
    let mut reports = Vec::new();
    for (index, (source, target)) in reference.iter().zip(actual).enumerate() {
        if source.tokens != target.tokens || source.logits.len() != target.logits.len() {
            return Err("validation input/vocabulary changed".into());
        }
        let (a, b) = (log_sum(&source.logits), log_sum(&target.logits));
        let kl = source
            .logits
            .iter()
            .zip(&target.logits)
            .map(|(&x, &y)| {
                let log_p = f64::from(x) - a;
                log_p.exp() * (log_p - (f64::from(y) - b))
            })
            .sum::<f64>();
        let top = |values: &[f32]| {
            let mut ids: Vec<_> = (0..values.len()).collect();
            ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            ids.truncate(10.min(ids.len()));
            ids
        };
        let left = top(&source.logits);
        let right = top(&target.logits);
        let overlap = left.iter().filter(|id| right.contains(id)).count();
        if !kl.is_finite() || kl < -1e-9 || kl > maximum_kl || overlap < minimum_top {
            return Err(format!(
                "holdout sample {} failed quality gates: KL={kl}, top10_overlap={overlap}",
                index + 1
            ));
        }
        reports.push(
            json!({"sample":index,"kl":kl.max(0.0),"top10_overlap":overlap,
            "source_generated":source.generated,"quantized_generated":target.generated,
            "forward_sha256":target.forward_sha256}),
        );
    }
    Ok(reports)
}

pub(super) fn verify_parity(expected: &[Snapshot], actual: &[Snapshot]) -> Result<(), String> {
    if expected.len() != actual.len()
        || expected.iter().zip(actual).any(|(a, b)| {
            a.tokens != b.tokens
                || a.forward_sha256 != b.forward_sha256
                || a.generated != b.generated
        })
    {
        return Err("published model differs from the in-memory quantized model".into());
    }
    Ok(())
}

pub(super) fn tensor_hash(tensor: &Tensor) -> Result<String, String> {
    let mut hash = Sha256::new();
    for value in tensor.as_f32().map_err(|e| e.to_string())? {
        hash.update(value.to_le_bytes());
    }
    Ok(hex(hash))
}

fn hex(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_overlap_threshold_is_not_lowered_for_a_small_vocabulary() {
        let sample = || Snapshot {
            tokens: vec![0],
            logits: vec![1.0, 0.0],
            forward_sha256: String::new(),
            generated: Vec::new(),
        };
        assert!(compare(&[sample()], &[sample()], 0.5, 3).is_err());
        assert!(compare(&[sample()], &[sample()], 0.5, 2).is_ok());
    }
}
