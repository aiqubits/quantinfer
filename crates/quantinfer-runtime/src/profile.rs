use serde::{Deserialize, Serialize};

use crate::{Result, RuntimeError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub weight_bytes: usize,
    pub activation_bytes: usize,
    pub temporary_workspace_bytes: usize,
    pub kv_cache_bytes: usize,
    pub output_logits_bytes: usize,
    pub maximum_context_length: usize,
}

impl MemoryPlan {
    pub fn for_model(
        weight_bytes: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_layers: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        maximum_context_length: usize,
    ) -> Result<Self> {
        if hidden_size == 0
            || vocab_size == 0
            || num_layers == 0
            || num_key_value_heads == 0
            || head_dim == 0
            || maximum_context_length == 0
        {
            return Err(RuntimeError::InvalidInput(
                "memory plan dimensions must be greater than zero".to_owned(),
            ));
        }
        let kv_cache_bytes = num_layers
            .checked_mul(maximum_context_length)
            .and_then(|bytes| bytes.checked_mul(num_key_value_heads))
            .and_then(|bytes| bytes.checked_mul(head_dim))
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                RuntimeError::InvalidInput("KV cache memory size overflows usize".to_owned())
            })?;
        let activation_bytes = maximum_context_length
            .checked_mul(hidden_size)
            .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                RuntimeError::InvalidInput("activation memory size overflows usize".to_owned())
            })?;
        let output_logits_bytes = vocab_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                RuntimeError::InvalidInput("logits memory size overflows usize".to_owned())
            })?;
        Ok(Self {
            weight_bytes,
            activation_bytes,
            temporary_workspace_bytes: activation_bytes,
            kv_cache_bytes,
            output_logits_bytes,
            maximum_context_length,
        })
    }

    pub fn total_bytes(&self) -> Result<usize> {
        [
            self.weight_bytes,
            self.activation_bytes,
            self.temporary_workspace_bytes,
            self.kv_cache_bytes,
            self.output_logits_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                RuntimeError::InvalidInput("memory plan total size overflows usize".to_owned())
            })
        })
    }

    pub fn ensure_budget(&self, budget_bytes: usize) -> Result<()> {
        let total = self.total_bytes()?;
        if total > budget_bytes {
            return Err(RuntimeError::InvalidInput(format!(
                "memory plan requires {total} bytes, budget is {budget_bytes}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceProfile {
    pub model_variant: String,
    pub quantization_variant: String,
    pub runtime_provider: String,
    pub kernel_provider: String,
    pub context_length: usize,
    pub thread_count: usize,
    pub memory_budget_bytes: Option<usize>,
    pub quality_threshold: Option<f32>,
    pub performance_baseline_tokens_per_second: Option<f64>,
}

impl InferenceProfile {
    pub fn validate(&self) -> Result<()> {
        if self.model_variant.is_empty()
            || self.quantization_variant.is_empty()
            || self.runtime_provider.is_empty()
            || self.kernel_provider.is_empty()
            || self.context_length == 0
            || self.thread_count == 0
        {
            return Err(RuntimeError::InvalidInput(
                "inference profile identity and resource fields must be valid".to_owned(),
            ));
        }
        if self
            .quality_threshold
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(RuntimeError::InvalidInput(
                "quality threshold must be finite and non-negative".to_owned(),
            ));
        }
        if self
            .performance_baseline_tokens_per_second
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(RuntimeError::InvalidInput(
                "performance baseline must be finite and non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InferenceProfile, MemoryPlan};

    #[test]
    fn computes_kv_cache_and_checks_budget() {
        let plan = MemoryPlan::for_model(1024, 8, 64, 2, 1, 4, 16).unwrap();
        assert_eq!(plan.kv_cache_bytes, 1024);
        assert!(plan.ensure_budget(plan.total_bytes().unwrap()).is_ok());
        assert!(plan.ensure_budget(plan.total_bytes().unwrap() - 1).is_err());
    }

    #[test]
    fn validates_inference_profile() {
        let profile = InferenceProfile {
            model_variant: "fixture".to_owned(),
            quantization_variant: "q8_0".to_owned(),
            runtime_provider: "cpu-scalar".to_owned(),
            kernel_provider: "cpu-scalar".to_owned(),
            context_length: 128,
            thread_count: 1,
            memory_budget_bytes: Some(1 << 20),
            quality_threshold: Some(0.99),
            performance_baseline_tokens_per_second: Some(10.0),
        };
        profile.validate().unwrap();
    }
}
