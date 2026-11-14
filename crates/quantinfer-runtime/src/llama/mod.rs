use quantinfer_core::{DType, Shape, Tensor};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::{LlamaConfig, TensorRole};

use crate::batch::{last_hidden_rows, DenseAttention, DenseBatchLimits};
use crate::calibration::{ProjectionObserver, ProjectionTrace};
use crate::linear::{add, LinearWeight};
use crate::{DecodeBatchItem, KvCache, LoraAdapter, LoraProjection, Result, RuntimeError};

#[derive(Debug, Clone)]
pub struct LlamaLayerWeights {
    pub input_layernorm: Tensor,
    pub q_proj: LinearWeight,
    pub k_proj: LinearWeight,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub post_attention_layernorm: Tensor,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

impl LlamaLayerWeights {
    pub fn validate(&self, config: &LlamaConfig) -> Result<()> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query = config.query_projection_size();
        let key_value = config.key_value_projection_size();
        let expected = [
            (
                "input_layernorm",
                self.input_layernorm.shape().dims(),
                &[hidden][..],
            ),
            ("q_proj", self.q_proj.shape().dims(), &[hidden, query][..]),
            (
                "k_proj",
                self.k_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            (
                "v_proj",
                self.v_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            ("o_proj", self.o_proj.shape().dims(), &[query, hidden][..]),
            (
                "post_attention_layernorm",
                self.post_attention_layernorm.shape().dims(),
                &[hidden][..],
            ),
            (
                "gate_proj",
                self.gate_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "up_proj",
                self.up_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "down_proj",
                self.down_proj.shape().dims(),
                &[intermediate, hidden][..],
            ),
        ];
        for (name, actual, expected) in expected {
            if actual != expected {
                return Err(RuntimeError::InvalidInput(format!(
                    "Llama {name} has shape {actual:?}, expected {expected:?}"
                )));
            }
            if !matches!(
                name,
                "q_proj" | "k_proj" | "v_proj" | "o_proj" | "gate_proj" | "up_proj" | "down_proj"
            ) && self.weight_dtype(name)? != DType::F32
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "Llama {name} must use F32 for the reference runtime"
                )));
            }
        }
        Ok(())
    }

    fn weight_dtype(&self, name: &str) -> Result<DType> {
        match name {
            "input_layernorm" => Ok(self.input_layernorm.dtype()),
            "q_proj" => Ok(self.q_proj.dtype()),
            "k_proj" => Ok(self.k_proj.dtype()),
            "v_proj" => Ok(self.v_proj.dtype()),
            "o_proj" => Ok(self.o_proj.dtype()),
            "post_attention_layernorm" => Ok(self.post_attention_layernorm.dtype()),
            "gate_proj" => Ok(self.gate_proj.dtype()),
            "up_proj" => Ok(self.up_proj.dtype()),
            "down_proj" => Ok(self.down_proj.dtype()),
            _ => Err(RuntimeError::InvalidInput(format!(
                "unknown Llama weight {name}"
            ))),
        }
    }

    fn forward<B: KernelBackend>(
        &self,
        backend: &B,
        config: &LlamaConfig,
        hidden: &Tensor,
        mut trace: ProjectionTrace<'_, '_>,
        attention: &DenseAttention,
        batch: &mut [DecodeBatchItem<'_>],
    ) -> Result<Tensor> {
        let sequence_length = hidden.shape().dims()[0];
        let residual = hidden.clone();
        let normalized = backend.rms_norm(hidden, &self.input_layernorm, config.rms_norm_eps)?;
        trace.record(
            &[
                TensorRole::QueryProjection,
                TensorRole::KeyProjection,
                TensorRole::ValueProjection,
            ],
            &normalized,
        )?;
        let query = self
            .q_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_attention_heads,
                config.effective_head_dim(),
            ]))?;
        let key = self
            .k_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads(),
                config.effective_head_dim(),
            ]))?;
        let value = self
            .v_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads(),
                config.effective_head_dim(),
            ]))?;
        let attended = attention
            .apply(backend, &query, &key, &value, trace.layer, batch)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.query_projection_size(),
            ]))?;
        trace.record(&[TensorRole::OutputProjection], &attended)?;
        let projected = self.o_proj.matmul(backend, &attended)?;
        let residual = add(&residual, &projected)?;
        let normalized = backend.rms_norm(
            &residual,
            &self.post_attention_layernorm,
            config.rms_norm_eps,
        )?;
        trace.record(
            &[TensorRole::GateProjection, TensorRole::UpProjection],
            &normalized,
        )?;
        let gate = self.gate_proj.matmul(backend, &normalized)?;
        let up = self.up_proj.matmul(backend, &normalized)?;
        let feed_forward = backend.silu_mul(&gate, &up)?;
        trace.record(&[TensorRole::DownProjection], &feed_forward)?;
        let feed_forward = self.down_proj.matmul(backend, &feed_forward)?;
        add(&residual, &feed_forward)
    }
}

pub struct LlamaModel<B> {
    pub config: LlamaConfig,
    backend: B,
    pub embedding: Tensor,
    pub layers: Vec<LlamaLayerWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
    lora_head: Option<LoraProjection>,
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected || tensor.dtype() != DType::F32 {
        return Err(RuntimeError::InvalidInput(format!(
            "Llama {name} has shape {:?} and dtype {:?}, expected shape {expected:?} F32",
            tensor.shape().dims(),
            tensor.dtype()
        )));
    }
    Ok(())
}

impl<B: KernelBackend> LlamaModel<B> {
    pub fn new(
        config: LlamaConfig,
        backend: B,
        embedding: Tensor,
        layers: Vec<LlamaLayerWeights>,
        final_norm: Tensor,
        lm_head: Tensor,
    ) -> Result<Self> {
        config.validate().map_err(RuntimeError::InvalidInput)?;
        expect_shape(
            &embedding,
            &[config.vocab_size, config.hidden_size],
            "embedding",
        )?;
        expect_shape(&final_norm, &[config.hidden_size], "final_norm")?;
        expect_shape(
            &lm_head,
            &[config.hidden_size, config.vocab_size],
            "lm_head",
        )?;
        if layers.len() != config.num_hidden_layers {
            return Err(RuntimeError::InvalidInput(format!(
                "Llama has {} layers but received {} weight sets",
                config.num_hidden_layers,
                layers.len()
            )));
        }
        for layer in &layers {
            layer.validate(&config)?;
        }
        Ok(Self {
            config,
            backend,
            embedding,
            layers,
            final_norm,
            lm_head,
            lora_head: None,
        })
    }

    /// Attach low-rank weights before populating a KV cache for the adapted model.
    pub fn attach_lora(&mut self, adapter: LoraAdapter) -> Result<()> {
        let mut projections = self
            .layers
            .iter_mut()
            .map(|layer| {
                [
                    &mut layer.q_proj,
                    &mut layer.k_proj,
                    &mut layer.v_proj,
                    &mut layer.o_proj,
                    &mut layer.gate_proj,
                    &mut layer.up_proj,
                    &mut layer.down_proj,
                ]
            })
            .collect::<Vec<_>>();
        adapter.install(&mut projections, &self.lm_head, &mut self.lora_head)
    }

    pub fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.forward_sequence(token_ids, cache, None)?;
        self.project_logits(&hidden)
    }

    /// Observe layer projection inputs while advancing the supplied KV cache.
    pub fn observe_projections(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
        observer: &mut dyn ProjectionObserver,
    ) -> Result<()> {
        self.forward_sequence(token_ids, cache, Some(observer))
            .map(drop)
    }

    fn forward_sequence(
        &self,
        token_ids: &[usize],
        cache: &mut KvCache,
        observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(RuntimeError::InvalidInput(
                "Llama forward requires at least one token".to_owned(),
            ));
        }
        let requested_length = cache
            .sequence_length(0)?
            .checked_add(token_ids.len())
            .ok_or_else(|| {
                RuntimeError::InvalidInput("Llama sequence length overflows usize".to_owned())
            })?;
        if requested_length > self.config.max_position_embeddings {
            return Err(RuntimeError::InvalidInput(
                "input exceeds Llama max_position_embeddings".to_owned(),
            ));
        }
        let mut request = [DecodeBatchItem {
            tokens: token_ids,
            cache,
        }];
        self.forward_hidden(token_ids, &mut request, observer)
    }

    pub fn forward_batch(&self, batch: &mut [DecodeBatchItem<'_>]) -> Vec<Result<Tensor>> {
        let limits = DenseBatchLimits {
            vocab_size: self.config.vocab_size,
            max_positions: self.config.max_position_embeddings,
            layers: self.layers.len(),
            key_value_heads: self.config.num_key_value_heads(),
            head_dim: self.config.effective_head_dim(),
        };
        limits.run(batch, |tokens, requests| {
            let hidden = self.forward_hidden(tokens, requests, None)?;
            self.project_logits(&last_hidden_rows(&hidden, requests)?)
        })
    }

    fn forward_hidden(
        &self,
        token_ids: &[usize],
        batch: &mut [DecodeBatchItem<'_>],
        mut observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor> {
        let attention = DenseAttention {
            query_heads: self.config.num_attention_heads,
            key_value_heads: self.config.num_key_value_heads(),
            head_dim: self.config.effective_head_dim(),
            rope_theta: self.config.rope_theta,
        };
        let mut hidden = self.backend.embedding(&self.embedding, token_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &self.backend,
                &self.config,
                &hidden,
                ProjectionTrace {
                    layer: index,
                    observer: &mut observer,
                },
                &attention,
                batch,
            )?;
        }
        Ok(hidden)
    }

    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self
            .backend
            .rms_norm(hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let logits = self.backend.matmul(&hidden, &self.lm_head)?;
        match &self.lora_head {
            Some(adapter) => adapter.add_to(&self.backend, &hidden, logits),
            None => Ok(logits),
        }
    }
}

#[cfg(test)]
mod tests;

mod loader;
pub use loader::uses_ggml_layout;
