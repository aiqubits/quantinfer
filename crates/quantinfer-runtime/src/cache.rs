//! Request-local key/value cache storage.

use quantinfer_core::{Shape, Tensor};

use crate::{Result, RuntimeError};

#[derive(Debug, Clone)]
pub(crate) struct LayerKvCache {
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
    pub(crate) sequence_length: usize,
}

/// Per-layer KV storage and attention scratch space for single-sequence inference.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub(crate) layers: Vec<LayerKvCache>,
    pub(crate) attention_scores: Vec<f32>,
    pub(crate) max_sequence_length: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
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
        let new_length = layer_cache
            .sequence_length
            .checked_add(key_dims[0])
            .ok_or_else(|| {
                RuntimeError::InvalidInput("KV cache sequence length overflows usize".to_owned())
            })?;
        if new_length > self.max_sequence_length {
            return Err(RuntimeError::InvalidInput(
                "KV cache capacity is smaller than the requested sequence".to_owned(),
            ));
        }
        let key_values = key.as_f32()?;
        let value_values = value.as_f32()?;
        layer_cache.keys.extend_from_slice(key_values);
        layer_cache.values.extend_from_slice(value_values);
        layer_cache.sequence_length = new_length;
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
