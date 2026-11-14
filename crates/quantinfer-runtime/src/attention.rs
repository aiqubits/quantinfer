//! Shared dense attention helpers.

use quantinfer_core::{Shape, Tensor};

use crate::{KvCache, Result, RuntimeError};

pub(crate) fn rope_tables(
    theta: f32,
    head_dim: usize,
    start: usize,
    length: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || !theta.is_finite() || theta <= 0.0 {
        return Err(RuntimeError::InvalidInput(
            "invalid RoPE dimensions or frequency base".to_owned(),
        ));
    }
    let pair_count = head_dim / 2;
    let entries = length
        .checked_mul(pair_count)
        .ok_or_else(|| RuntimeError::InvalidInput("RoPE table size overflows usize".to_owned()))?;
    let end = start.checked_add(length).ok_or_else(|| {
        RuntimeError::InvalidInput("RoPE position range overflows usize".to_owned())
    })?;
    let mut cos = Vec::with_capacity(entries);
    let mut sin = Vec::with_capacity(entries);
    for position in start..end {
        for pair in 0..pair_count {
            let exponent = (2.0 * pair as f32) / head_dim as f32;
            let angle = position as f32 * theta.powf(-exponent);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((cos, sin))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attention(
    query: &Tensor,
    cache: &mut KvCache,
    layer_index: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    start_position: usize,
    reuse_workspace: bool,
) -> Result<Tensor> {
    let layer = cache
        .layers
        .get(layer_index)
        .ok_or_else(|| RuntimeError::InvalidInput("KV layer index is out of bounds".to_owned()))?;
    let key_values = &layer.keys;
    let value_values = &layer.values;
    let key_tokens = layer.sequence_length;
    let query_dims = query.shape().dims();
    if query_dims.len() != 3
        || query_dims[1] != query_heads
        || query_dims[2] != head_dim
        || key_tokens == 0
        || key_value_heads == 0
        || key_tokens
            .checked_mul(key_value_heads)
            .and_then(|size| size.checked_mul(head_dim))
            != Some(key_values.len())
        || value_values.len() != key_values.len()
        || !query_heads.is_multiple_of(key_value_heads)
    {
        return Err(RuntimeError::InvalidInput(
            "invalid Qwen3 GQA tensor shapes".to_owned(),
        ));
    }
    let query_values = query.as_f32()?;
    let query_tokens = query_dims[0];
    let query_end = start_position.checked_add(query_tokens).ok_or_else(|| {
        RuntimeError::InvalidInput("attention position range overflows usize".to_owned())
    })?;
    if key_tokens < query_end {
        return Err(RuntimeError::InvalidInput(
            "KV cache does not contain all causal attention keys".to_owned(),
        ));
    }
    let group_size = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let output_elements = query_tokens
        .checked_mul(query_heads)
        .and_then(|size| size.checked_mul(head_dim))
        .ok_or_else(|| {
            RuntimeError::InvalidInput("attention output size overflows usize".to_owned())
        })?;
    let mut output = vec![0.0; output_elements];
    let mut local_scores = Vec::new();
    let scores = if reuse_workspace {
        &mut cache.attention_scores
    } else {
        &mut local_scores
    };
    scores.clear();
    if scores.capacity() < key_tokens {
        let target = key_tokens
            .max(scores.capacity().saturating_mul(2))
            .min(cache.max_sequence_length);
        scores.reserve_exact(target);
    }
    for token in 0..query_tokens {
        let last_key = start_position + token;
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = (token * query_heads + query_head) * head_dim;
            scores.clear();
            for key_token in 0..=last_key {
                let key_start = (key_token * key_value_heads + kv_head) * head_dim;
                let score = query_values[query_start..query_start + head_dim]
                    .iter()
                    .zip(&key_values[key_start..key_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
            let output_start = query_start;
            for (key_token, score) in scores.iter().copied().enumerate() {
                let weight = (score - maximum).exp() / denominator;
                let value_start = (key_token * key_value_heads + kv_head) * head_dim;
                for dimension in 0..head_dim {
                    output[output_start + dimension] +=
                        weight * value_values[value_start + dimension];
                }
            }
        }
    }
    Ok(Tensor::from_f32(
        Shape::from(vec![query_tokens, query_heads, head_dim]),
        output,
    )?)
}
