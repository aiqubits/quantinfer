//! Token selection and streaming generation.

use quantinfer_core::Tensor;

use crate::{DecoderModel, KvCache, Qwen3Model, Result, RuntimeError, StreamControl, TokenSink};

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub eos_token_id: Option<usize>,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            eos_token_id: None,
            seed: 0,
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_new_tokens == 0 {
            return Err(RuntimeError::InvalidInput(
                "max_new_tokens must be greater than zero".to_owned(),
            ));
        }
        if self.temperature < 0.0 || !self.temperature.is_finite() {
            return Err(RuntimeError::InvalidInput(
                "temperature must be finite and non-negative".to_owned(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(RuntimeError::InvalidInput(
                "top_k must be greater than zero when provided".to_owned(),
            ));
        }
        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) || !top_p.is_finite() || top_p == 0.0 {
                return Err(RuntimeError::InvalidInput(
                    "top_p must be finite and in the range (0, 1]".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

pub fn argmax(logits: &Tensor) -> Result<usize> {
    let values = logits.as_f32()?;
    if values.is_empty() {
        return Err(RuntimeError::InvalidInput(
            "logits cannot be empty".to_owned(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::InvalidInput(
            "logits must contain only finite values".to_owned(),
        ));
    }
    let mut best_index = 0;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > values[best_index] {
            best_index = index;
        }
    }
    Ok(best_index)
}

pub fn select_token(logits: &Tensor, config: &GenerationConfig) -> Result<usize> {
    config.validate()?;
    let values = logits.as_f32()?;
    if values.is_empty() {
        return Err(RuntimeError::InvalidInput(
            "logits cannot be empty".to_owned(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::InvalidInput(
            "logits must contain only finite values".to_owned(),
        ));
    }
    if config.temperature == 0.0 && config.top_k.is_none() && config.top_p.is_none() {
        return argmax(logits);
    }

    let temperature = if config.temperature == 0.0 {
        1.0
    } else {
        config.temperature
    };
    let mut candidates: Vec<(usize, f32)> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index, value / temperature))
        .collect();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if let Some(top_k) = config.top_k {
        candidates.truncate(top_k.min(candidates.len()));
    }
    let maximum = candidates[0].1;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|(_, value)| (*value - maximum).exp())
        .collect();
    if let Some(top_p) = config.top_p {
        let total = weights.iter().sum::<f32>();
        let mut cumulative = 0.0;
        let mut kept = 0;
        for weight in &weights {
            cumulative += *weight;
            kept += 1;
            if cumulative / total >= top_p {
                break;
            }
        }
        candidates.truncate(kept.max(1));
    }
    if config.temperature == 0.0 {
        return candidates.first().map(|(index, _)| *index).ok_or_else(|| {
            RuntimeError::InvalidInput("sampling candidate set is empty".to_owned())
        });
    }
    let maximum = candidates[0].1;
    let weights: Vec<f32> = candidates
        .iter()
        .map(|(_, value)| (*value - maximum).exp())
        .collect();
    let total = weights.iter().sum::<f32>();
    let target = deterministic_unit(config.seed, values);
    let mut cumulative = 0.0;
    for ((index, _), weight) in candidates.iter().zip(weights) {
        cumulative += weight / total;
        if target <= cumulative {
            return Ok(*index);
        }
    }
    candidates
        .last()
        .map(|(index, _)| *index)
        .ok_or_else(|| RuntimeError::InvalidInput("sampling candidate set is empty".to_owned()))
}

pub fn generate_qwen3_stream<B, S>(
    model: &Qwen3Model<B>,
    prompt_tokens: &[usize],
    cache: &mut KvCache,
    config: &GenerationConfig,
    sink: &mut S,
) -> Result<Vec<usize>>
where
    B: quantinfer_kernel::KernelBackend,
    S: TokenSink,
{
    generate_stream(model, prompt_tokens, cache, config, sink)
}

/// Emit selected token IDs from either supported decoder, using the supplied KV cache.
pub fn generate_stream<M, S>(
    model: &M,
    prompt_tokens: &[usize],
    cache: &mut KvCache,
    config: &GenerationConfig,
    sink: &mut S,
) -> Result<Vec<usize>>
where
    M: DecoderModel + ?Sized,
    S: TokenSink + ?Sized,
{
    config.validate()?;
    if prompt_tokens.is_empty() {
        return Err(RuntimeError::InvalidInput(
            "streaming generation requires a non-empty prompt".to_owned(),
        ));
    }
    let mut logits = model.forward(prompt_tokens, cache)?;
    let mut generated = Vec::with_capacity(config.max_new_tokens);
    for _ in 0..config.max_new_tokens {
        let next = select_token(&last_logits(&logits)?, config)?;
        generated.push(next);
        let control = sink.on_token(next)?;
        if config.eos_token_id == Some(next) || control == StreamControl::Stop {
            break;
        }
        logits = model.forward(&[next], cache)?;
    }
    Ok(generated)
}

pub(crate) fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(RuntimeError::InvalidInput(
            "model logits must have shape [tokens, vocab]".to_owned(),
        ));
    }
    let values = logits.as_f32()?;
    let vocab_size = dims[1];
    let start = (dims[0] - 1) * vocab_size;
    Ok(Tensor::from_f32(
        vec![vocab_size],
        values[start..].to_vec(),
    )?)
}

fn deterministic_unit(seed: u64, values: &[f32]) -> f32 {
    let mut state = seed ^ values.len() as u64;
    for value in values {
        state ^= u64::from(value.to_bits());
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state = state.rotate_left(17);
    }
    let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
    unit.min(1.0 - f32::EPSILON)
}
