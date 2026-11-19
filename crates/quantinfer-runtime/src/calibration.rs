use quantinfer_core::Tensor;
use quantinfer_model::TensorRole;

use crate::batch::DenseAttention;
use crate::{KvCache, Result, RuntimeError};

pub(crate) fn validate_block_input(input: &Tensor, width: usize) -> Result<usize> {
    let &[rows, channels] = input.shape().dims() else {
        return Err(RuntimeError::InvalidInput(
            "block input must have shape [tokens,hidden]".into(),
        ));
    };
    if rows == 0 || channels != width || input.as_f32()?.iter().any(|v| !v.is_finite()) {
        return Err(RuntimeError::InvalidInput(
            "block input dimensions or finite values are invalid".into(),
        ));
    }
    Ok(rows)
}

pub(crate) fn validate_block_cache(
    cache: &KvCache,
    layer: usize,
    rows: usize,
    layers: usize,
    attention: &DenseAttention,
    maximum_positions: usize,
) -> Result<()> {
    if cache.layers.len() != layers
        || cache.num_key_value_heads != attention.key_value_heads
        || cache.head_dim != attention.head_dim
    {
        return Err(RuntimeError::InvalidInput(
            "block KV layout differs from model".into(),
        ));
    }
    let end = cache
        .sequence_length(layer)?
        .checked_add(rows)
        .ok_or_else(|| RuntimeError::InvalidInput("block sequence length overflows".into()))?;
    if end > maximum_positions.min(cache.max_sequence_length) {
        return Err(RuntimeError::InvalidInput(
            "block input exceeds model or KV capacity".into(),
        ));
    }
    Ok(())
}

/// Receives a [tokens, channels] activation matrix before a linear projection.
/// Roles that consume the same activations are reported together.
pub trait ProjectionObserver {
    fn observe(&mut self, layer: usize, roles: &[TensorRole], activations: &Tensor) -> Result<()>;
}

impl<F> ProjectionObserver for F
where
    F: FnMut(usize, &[TensorRole], &Tensor) -> Result<()>,
{
    fn observe(&mut self, layer: usize, roles: &[TensorRole], activations: &Tensor) -> Result<()> {
        self(layer, roles, activations)
    }
}

pub(crate) struct ProjectionTrace<'a, 'observer> {
    pub layer: usize,
    pub observer: &'a mut Option<&'observer mut dyn ProjectionObserver>,
}

impl ProjectionTrace<'_, '_> {
    pub fn record(&mut self, roles: &[TensorRole], activations: &Tensor) -> Result<()> {
        if let Some(observer) = self.observer {
            observer.observe(self.layer, roles, activations)?;
        }
        Ok(())
    }
}
