use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;

use crate::qwen3::{gqa_attention, rope_tables};
use crate::{KvCache, Result, RuntimeError};

/// A sequence's input tokens and its private attention state for one batch step.
pub struct DecodeBatchItem<'a> {
    pub tokens: &'a [usize],
    pub cache: &'a mut KvCache,
}

pub(crate) struct DenseAttention {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl DenseAttention {
    pub fn apply(
        &self,
        backend: &impl KernelBackend,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        layer: usize,
        batch: &mut [DecodeBatchItem<'_>],
    ) -> Result<Tensor> {
        let tables = |request: &DecodeBatchItem<'_>| {
            let start = request.cache.sequence_length(layer)?;
            Ok::<_, RuntimeError>(rope_tables(
                self.rope_theta,
                self.head_dim,
                start,
                request.tokens.len(),
            ))
        };
        let (cos, sin) = if let [request] = batch {
            tables(request)?
        } else {
            let entries = query.shape().dims()[0] * (self.head_dim / 2);
            let mut cos = Vec::with_capacity(entries);
            let mut sin = Vec::with_capacity(entries);
            for request in batch.iter() {
                let (request_cos, request_sin) = tables(request)?;
                cos.extend(request_cos);
                sin.extend(request_sin);
            }
            (cos, sin)
        };
        let query = backend.apply_rope(query, &cos, &sin)?;
        let key = backend.apply_rope(key, &cos, &sin)?;
        let attend = |query: &Tensor, key: &Tensor, value: &Tensor, cache: &mut KvCache| {
            let start = cache.sequence_length(layer)?;
            cache.append(layer, key, value)?;
            gqa_attention(
                query,
                cache,
                layer,
                self.query_heads,
                self.key_value_heads,
                self.head_dim,
                start,
                backend.reuse_workspace(),
            )
        };
        if let [request] = batch {
            return attend(&query, &key, value, request.cache);
        }

        let mut output =
            Vec::with_capacity(query.shape().num_elements().map_err(RuntimeError::Core)?);
        let mut offset = 0;
        for request in batch {
            let count = request.tokens.len();
            let attended = attend(
                &token_rows(&query, offset, count)?,
                &token_rows(&key, offset, count)?,
                &token_rows(value, offset, count)?,
                request.cache,
            )?;
            output.extend_from_slice(attended.as_f32()?);
            offset += count;
        }
        Ok(Tensor::from_f32(query.shape().clone(), output)?)
    }
}

fn token_rows(tensor: &Tensor, offset: usize, count: usize) -> Result<Tensor> {
    let &[tokens, heads, head_dim] = tensor.shape().dims() else {
        return Err(RuntimeError::InvalidInput(
            "invalid batched attention shape".to_owned(),
        ));
    };
    if offset > tokens || count > tokens - offset {
        return Err(RuntimeError::InvalidInput(
            "batched attention rows are missing".to_owned(),
        ));
    }
    let width = heads * head_dim;
    let values = &tensor.as_f32()?[offset * width..(offset + count) * width];
    Ok(Tensor::from_f32(
        vec![count, heads, head_dim],
        values.to_vec(),
    )?)
}

pub(crate) struct DenseBatchLimits {
    pub vocab_size: usize,
    pub max_positions: usize,
    pub layers: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
}

impl DenseBatchLimits {
    fn check(&self, request: &DecodeBatchItem<'_>) -> Result<usize> {
        let cache = &request.cache;
        if request.tokens.is_empty() || request.tokens.iter().any(|&id| id >= self.vocab_size) {
            return Err(RuntimeError::InvalidInput(
                "invalid batch input tokens".to_owned(),
            ));
        }
        if cache.layers.len() != self.layers
            || cache.num_key_value_heads != self.key_value_heads
            || cache.head_dim != self.head_dim
        {
            return Err(RuntimeError::InvalidInput(
                "batch KV layout differs from model".to_owned(),
            ));
        }
        let start = cache.sequence_length(0)?;
        if cache
            .layers
            .iter()
            .any(|layer| layer.sequence_length != start)
        {
            return Err(RuntimeError::InvalidInput(
                "batch KV layers have different lengths".to_owned(),
            ));
        }
        if request.tokens.len()
            > self
                .max_positions
                .min(cache.max_sequence_length)
                .saturating_sub(start)
        {
            return Err(RuntimeError::InvalidInput(
                "batch input exceeds sequence capacity".to_owned(),
            ));
        }
        Ok(start)
    }

    pub fn run(
        &self,
        requests: &mut [DecodeBatchItem<'_>],
        forward: impl FnOnce(&[usize], &mut [DecodeBatchItem<'_>]) -> Result<Tensor>,
    ) -> Vec<Result<Tensor>> {
        let mut accepted = Vec::with_capacity(requests.len());
        let mut starts = Vec::with_capacity(requests.len());
        let mut tokens = Vec::new();
        let mut rejected = Vec::with_capacity(requests.len());
        for request in requests {
            match self.check(request) {
                Ok(start) => {
                    tokens.extend_from_slice(request.tokens);
                    accepted.push(DecodeBatchItem {
                        tokens: request.tokens,
                        cache: request.cache,
                    });
                    starts.push(start);
                    rejected.push(None);
                }
                Err(error) => rejected.push(Some(error)),
            }
        }
        if accepted.is_empty() {
            return rejected.into_iter().flatten().map(Err).collect();
        }
        let batch_size = accepted.len();
        let outputs = forward(&tokens, &mut accepted).and_then(|logits| {
            if logits.shape().dims() != [batch_size, self.vocab_size] {
                return Err(RuntimeError::InvalidInput(
                    "invalid batched logits shape".to_owned(),
                ));
            }
            logits
                .as_f32()?
                .chunks_exact(self.vocab_size)
                .map(|row| {
                    Tensor::from_f32(vec![self.vocab_size], row.to_vec()).map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()
        });
        match outputs {
            Ok(outputs) => {
                let mut logits = outputs.into_iter();
                rejected
                    .into_iter()
                    .map(|error| match error {
                        Some(error) => Err(error),
                        None => Ok(logits.next().expect("one row for each accepted request")),
                    })
                    .collect()
            }
            Err(error) => {
                // A failure in a shared operator must leave each request's KV state retryable.
                for (request, start) in accepted.into_iter().zip(starts) {
                    let elements = start * self.key_value_heads * self.head_dim;
                    for layer in &mut request.cache.layers {
                        layer.keys.truncate(elements);
                        layer.values.truncate(elements);
                        layer.sequence_length = start;
                    }
                }
                rejected
                    .into_iter()
                    .map(|rejection| Err(rejection.unwrap_or_else(|| error.clone())))
                    .collect()
            }
        }
    }
}

pub(crate) fn last_hidden_rows(hidden: &Tensor, batch: &[DecodeBatchItem<'_>]) -> Result<Tensor> {
    let &[rows, width] = hidden.shape().dims() else {
        return Err(RuntimeError::InvalidInput(
            "invalid batched hidden shape".to_owned(),
        ));
    };
    if rows
        != batch
            .iter()
            .map(|request| request.tokens.len())
            .sum::<usize>()
    {
        return Err(RuntimeError::InvalidInput(
            "batched hidden rows are missing".to_owned(),
        ));
    }
    let values = hidden.as_f32()?;
    let mut output = Vec::with_capacity(batch.len() * width);
    let mut end = 0;
    for request in batch {
        end += request.tokens.len() * width;
        output.extend_from_slice(&values[end - width..end]);
    }
    Ok(Tensor::from_f32(vec![batch.len(), width], output)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolls_back_every_cache_after_a_shared_operator_error() {
        let mut left = KvCache::new(2, 4, 1, 2).unwrap();
        let mut right = left.clone();
        let kv = Tensor::from_f32(vec![1, 1, 2], vec![1.0, 2.0]).unwrap();
        for layer in 0..2 {
            left.append(layer, &kv, &kv).unwrap();
        }
        let mut requests = [
            DecodeBatchItem {
                tokens: &[0],
                cache: &mut left,
            },
            DecodeBatchItem {
                tokens: &[1],
                cache: &mut right,
            },
        ];
        let limits = DenseBatchLimits {
            vocab_size: 2,
            max_positions: 4,
            layers: 2,
            key_value_heads: 1,
            head_dim: 2,
        };
        let result = limits.run(&mut requests, |_, requests| {
            requests[0].cache.append(0, &kv, &kv)?;
            requests[1].cache.append(0, &kv, &kv)?;
            Err(RuntimeError::InvalidInput(
                "fixture operator failed".to_owned(),
            ))
        });
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|result| result.is_err()));
        for layer in 0..2 {
            assert_eq!(left.sequence_length(layer).unwrap(), 1);
            assert_eq!(left.key_values(layer).unwrap(), &[1.0, 2.0]);
            assert_eq!(right.sequence_length(layer).unwrap(), 0);
            assert!(right.key_values(layer).unwrap().is_empty());
        }
    }
}
