//! Runtime state shared by model execution and generation.

#![forbid(unsafe_code)]

pub mod llama;
pub mod profile;
pub mod provider;
pub mod qwen3;
pub mod scheduler;
pub mod session;

pub use llama::{LlamaLayerWeights, LlamaModel};
pub use profile::{InferenceProfile, MemoryPlan};
pub use provider::*;
pub use qwen3::{LinearWeight, Qwen3LayerWeights, Qwen3Model};
pub use scheduler::{DecodeEvent, DecodeScheduler};
pub use session::{DecodeSession, DecodeStopReason};

use std::fmt;

use quantinfer_core::{Shape, Tensor};
use quantinfer_model::{ModelAdapter, ModelSpec, TensorRole};

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeError {
    InvalidInput(String),
    Core(quantinfer_core::CoreError),
    Format(quantinfer_format::FormatError),
    Kernel(quantinfer_kernel::KernelError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamControl {
    Continue,
    Stop,
}

pub trait TokenSink {
    fn on_token(&mut self, token_id: usize) -> Result<StreamControl>;
}

/// Forward pass for a causal decoder with request-local KV state.
pub trait DecoderModel {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor>;
}

impl<B: quantinfer_kernel::KernelBackend> DecoderModel for Qwen3Model<B> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        Qwen3Model::forward(self, token_ids, cache)
    }
}

impl<B: quantinfer_kernel::KernelBackend> DecoderModel for LlamaModel<B> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        LlamaModel::forward(self, token_ids, cache)
    }
}

impl<F> TokenSink for F
where
    F: FnMut(usize) -> Result<StreamControl>,
{
    fn on_token(&mut self, token_id: usize) -> Result<StreamControl> {
        self(token_id)
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Core(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Kernel(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<quantinfer_core::CoreError> for RuntimeError {
    fn from(error: quantinfer_core::CoreError) -> Self {
        Self::Core(error)
    }
}

impl From<quantinfer_format::FormatError> for RuntimeError {
    fn from(error: quantinfer_format::FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<quantinfer_kernel::KernelError> for RuntimeError {
    fn from(error: quantinfer_kernel::KernelError) -> Self {
        Self::Kernel(error)
    }
}

fn tensor_source(spec: &ModelSpec, role: TensorRole, layer: Option<usize>) -> Result<&str> {
    spec.tensor(role, layer)
        .map(|tensor| tensor.source_name.as_str())
        .ok_or_else(|| {
            RuntimeError::InvalidInput(format!(
                "{} model spec is missing tensor role {role:?} for layer {layer:?}",
                spec.architecture,
            ))
        })
}

fn lm_head_source(adapter: &impl ModelAdapter) -> Result<String> {
    let spec = adapter.model_spec();
    if spec.embedding.tied_lm_head {
        // A tied checkpoint may still store an explicit output weight.
        adapter
            .canonical_tensor_name("lm_head.weight")
            .ok_or_else(|| {
                RuntimeError::InvalidInput(format!(
                    "{} adapter is missing the LM head mapping",
                    spec.architecture
                ))
            })
    } else {
        tensor_source(spec, TensorRole::LmHead, None).map(str::to_owned)
    }
}

#[derive(Debug, Clone)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    sequence_length: usize,
}

/// Per-layer KV storage and attention scratch space for single-sequence inference.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    attention_scores: Vec<f32>,
    max_sequence_length: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        max_sequence_length: usize,
        num_key_value_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if num_layers == 0 || max_sequence_length == 0 || num_key_value_heads == 0 || head_dim == 0
        {
            return Err(RuntimeError::InvalidInput(
                "KV cache dimensions must be greater than zero".to_owned(),
            ));
        }
        max_sequence_length
            .checked_mul(num_key_value_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(num_layers))
            .and_then(|value| value.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                RuntimeError::InvalidInput("KV cache dimensions overflow usize".to_owned())
            })?;
        Ok(Self {
            layers: (0..num_layers)
                .map(|_| LayerKvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                    sequence_length: 0,
                })
                .collect(),
            attention_scores: Vec::new(),
            max_sequence_length,
            num_key_value_heads,
            head_dim,
        })
    }

    pub fn sequence_length(&self, layer: usize) -> Result<usize> {
        self.layers
            .get(layer)
            .map(|cache| cache.sequence_length)
            .ok_or_else(|| RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned()))
    }

    pub fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        let layer_cache = self.layers.get_mut(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned())
        })?;
        let expected = [self.num_key_value_heads, self.head_dim];
        let key_dims = key.shape().dims();
        let value_dims = value.shape().dims();
        if key_dims.len() != 3
            || value_dims != key_dims
            || key_dims[1..] != expected
            || key_dims[0] == 0
        {
            return Err(RuntimeError::InvalidInput(
                "KV tensors must have shape [tokens, num_kv_heads, head_dim]".to_owned(),
            ));
        }
        if layer_cache.sequence_length + key_dims[0] > self.max_sequence_length {
            return Err(RuntimeError::InvalidInput(
                "KV cache capacity is smaller than the requested sequence".to_owned(),
            ));
        }
        let key_values = key.as_f32()?;
        let value_values = value.as_f32()?;
        layer_cache.keys.extend_from_slice(key_values);
        layer_cache.values.extend_from_slice(value_values);
        layer_cache.sequence_length += key_dims[0];
        Ok(())
    }

    pub fn keys(&self, layer: usize) -> Result<Tensor> {
        self.tensor_for(layer, true)
    }

    pub fn values(&self, layer: usize) -> Result<Tensor> {
        self.tensor_for(layer, false)
    }

    pub fn key_values(&self, layer: usize) -> Result<&[f32]> {
        let layer_cache = self.layers.get(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned())
        })?;
        Ok(&layer_cache.keys)
    }

    pub fn value_values(&self, layer: usize) -> Result<&[f32]> {
        let layer_cache = self.layers.get(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned())
        })?;
        Ok(&layer_cache.values)
    }

    fn tensor_for(&self, layer: usize, keys: bool) -> Result<Tensor> {
        let layer_cache = self.layers.get(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned())
        })?;
        let values = if keys {
            &layer_cache.keys
        } else {
            &layer_cache.values
        };
        Ok(Tensor::from_f32(
            Shape::from(vec![
                layer_cache.sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ]),
            values.clone(),
        )?)
    }

    pub fn clear(&mut self) {
        self.attention_scores.clear();
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
            layer.sequence_length = 0;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| (layer.keys.len() + layer.values.len()) * std::mem::size_of::<f32>())
            .sum()
    }

    pub fn capacity_bytes(&self) -> usize {
        self.layers.len()
            * self.max_sequence_length
            * self.num_key_value_heads
            * self.head_dim
            * 2
            * std::mem::size_of::<f32>()
    }
}

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

fn last_logits(logits: &Tensor) -> Result<Tensor> {
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

#[cfg(test)]
mod tests {
    use super::{argmax, select_token, GenerationConfig, KvCache};
    use quantinfer_core::Tensor;

    #[test]
    fn appends_and_reads_kv_cache() {
        let mut cache = KvCache::new(2, 8, 2, 4).unwrap();
        let key = Tensor::from_f32(vec![1, 2, 4], vec![1.0; 8]).unwrap();
        let value = Tensor::from_f32(vec![1, 2, 4], vec![2.0; 8]).unwrap();
        cache.append(0, &key, &value).unwrap();
        assert_eq!(cache.sequence_length(0).unwrap(), 1);
        assert_eq!(cache.capacity_bytes(), 2 * 8 * 2 * 4 * 2 * 4);
        assert!(cache.memory_bytes() < cache.capacity_bytes());
        assert_eq!(cache.key_values(0).unwrap(), &[1.0; 8]);
        assert_eq!(cache.value_values(0).unwrap(), &[2.0; 8]);
        assert_eq!(cache.keys(0).unwrap().as_f32().unwrap(), &[1.0; 8]);
        assert_eq!(cache.values(0).unwrap().as_f32().unwrap(), &[2.0; 8]);
    }

    #[test]
    fn selects_deterministic_highest_logit() {
        let logits = Tensor::from_f32(vec![4], vec![0.1, 0.9, 0.2, 0.8]).unwrap();
        assert_eq!(argmax(&logits).unwrap(), 1);
        assert_eq!(
            select_token(&logits, &GenerationConfig::default()).unwrap(),
            1
        );
    }
}
