//! Actual decoder components exposed for offline block calibration.
use super::*;
use crate::calibration::{validate_block_cache, validate_block_input};

impl<B: KernelBackend> LlamaModel<B> {
    pub(super) fn block_attention(&self) -> DenseAttention {
        DenseAttention {
            query_heads: self.config.num_attention_heads,
            key_value_heads: self.config.num_key_value_heads(),
            head_dim: self.config.effective_head_dim(),
            rope_theta: self.config.rope_theta,
        }
    }

    /// Evaluate attention including its output projection, from already
    /// normalized [tokens,hidden] activations. No residual or RMSNorm is applied.
    /// Advances only this layer's cache; other layers may have different lengths.
    pub fn forward_attention_block(
        &self,
        layer: usize,
        input: &Tensor,
        cache: &mut KvCache,
        mut observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        self.config.validate().map_err(RuntimeError::InvalidInput)?;
        let weights = self.layers.get(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("calibration layer is out of bounds".into())
        })?;
        weights.validate(&self.config)?;
        let rows = validate_block_input(input, self.config.hidden_size)?;
        let attention = self.block_attention();
        validate_block_cache(
            cache,
            layer,
            rows,
            self.layers.len(),
            &attention,
            self.config.max_position_embeddings,
        )?;
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(rows).map_err(|_| {
            RuntimeError::InvalidInput("cannot allocate calibration row labels".into())
        })?;
        tokens.resize(rows, 0);
        weights.attention_block(
            &self.backend,
            &self.config,
            input,
            &mut ProjectionTrace {
                layer,
                observer: &mut observer,
            },
            &attention,
            &mut [DecodeBatchItem {
                tokens: &tokens,
                cache,
            }],
        )
    }

    /// Evaluate the complete SwiGLU feed-forward block from already normalized
    /// activations. No residual or RMSNorm is applied and no KV state is changed.
    pub fn forward_feed_forward_block(
        &self,
        layer: usize,
        input: &Tensor,
        mut observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        self.config.validate().map_err(RuntimeError::InvalidInput)?;
        let weights = self.layers.get(layer).ok_or_else(|| {
            RuntimeError::InvalidInput("calibration layer is out of bounds".into())
        })?;
        weights.validate(&self.config)?;
        validate_block_input(input, self.config.hidden_size)?;
        weights.feed_forward_block(
            &self.backend,
            input,
            &mut ProjectionTrace {
                layer,
                observer: &mut observer,
            },
        )
    }
}
