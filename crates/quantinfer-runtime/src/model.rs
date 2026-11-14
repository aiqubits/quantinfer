//! Decoder model contracts and semantic tensor lookups.

use quantinfer_core::Tensor;
use quantinfer_model::{ModelSpec, TensorRole};

use crate::generation::last_logits;
use crate::storage::dense_storage_name;
use crate::{DecodeBatchItem, KvCache, LlamaModel, Qwen3Model, Result, RuntimeError};

/// Forward pass for a causal decoder with request-local KV state.
pub trait DecoderModel {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor>;

    /// Return one last-position logits vector per input sequence in request order.
    fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        batch
            .iter_mut()
            .map(|request| last_logits(&self.forward(request.tokens, request.cache)?))
            .collect()
    }
}

impl<B: quantinfer_kernel::KernelBackend> DecoderModel for Qwen3Model<B> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        Qwen3Model::forward(self, token_ids, cache)
    }

    fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        Qwen3Model::forward_batch(self, batch)
    }
}

impl<B: quantinfer_kernel::KernelBackend> DecoderModel for LlamaModel<B> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        LlamaModel::forward(self, token_ids, cache)
    }

    fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        LlamaModel::forward_batch(self, batch)
    }
}

pub(crate) fn tensor_source(
    spec: &ModelSpec,
    role: TensorRole,
    layer: Option<usize>,
) -> Result<String> {
    spec.tensor(role, layer)
        .map(|_| dense_storage_name(role, layer))
        .ok_or_else(|| {
            RuntimeError::InvalidInput(format!(
                "{} model spec is missing tensor role {role:?} for layer {layer:?}",
                spec.architecture,
            ))
        })
}

pub(crate) fn lm_head_source(spec: &ModelSpec) -> Result<String> {
    if spec.embedding.tied_lm_head {
        Ok(dense_storage_name(TensorRole::LmHead, None))
    } else {
        tensor_source(spec, TensorRole::LmHead, None)
    }
}
