use quantinfer_core::QuantizationScheme;
use quantinfer_format::{ExecutionVariant, ModelManifest};
use quantinfer_model::ModelSpec;
use serde::{Deserialize, Serialize};

use crate::{ExecutionPlan, Result, RuntimeError};

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
    pub fn for_dense_prefill(
        model: &ModelSpec,
        weight_bytes: usize,
        maximum_context_length: usize,
        prefill_tokens: usize,
    ) -> Result<Self> {
        if prefill_tokens == 0 || prefill_tokens > maximum_context_length {
            return Err(RuntimeError::InvalidInput(
                "prefill tokens must fit the memory plan context".to_owned(),
            ));
        }
        let attention = &model
            .layers
            .first()
            .ok_or_else(|| RuntimeError::InvalidInput("model has no layers".to_owned()))?
            .attention;
        let mut plan = Self::for_model(
            weight_bytes,
            model.hidden_size,
            model.vocabulary.size,
            model.layers.len(),
            attention.num_key_value_heads,
            attention.head_dim,
            maximum_context_length,
        )?;
        let allocation = |width: usize, buffers: usize| {
            prefill_tokens
                .checked_mul(width)
                .and_then(|count| count.checked_mul(buffers))
                .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    RuntimeError::InvalidInput("prefill memory size overflows usize".to_owned())
                })
        };
        plan.activation_bytes = allocation(model.hidden_size, 4)?;
        let projection_width = attention
            .num_heads
            .checked_mul(attention.head_dim)
            .ok_or_else(|| {
                RuntimeError::InvalidInput("attention width overflows usize".to_owned())
            })?;
        let attention_scores_bytes = maximum_context_length
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                RuntimeError::InvalidInput("attention scratch size overflows usize".to_owned())
            })?;
        plan.temporary_workspace_bytes = allocation(model.intermediate_size, 4)?
            .max(allocation(projection_width, 8)?)
            .checked_add(attention_scores_bytes)
            .ok_or_else(|| {
                RuntimeError::InvalidInput("workspace size overflows usize".to_owned())
            })?;
        plan.output_logits_bytes = allocation(model.vocabulary.size, 1)?;
        Ok(plan)
    }

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

    /// Match a package variant to the runtime and kernel capabilities selected for it.
    pub fn bind<'a>(
        &self,
        manifest: &'a ModelManifest,
        plan: &ExecutionPlan,
    ) -> Result<&'a ExecutionVariant> {
        self.validate()?;
        self.check_plan(plan)?;
        self.bind_variant(manifest, plan.capability.quantization)
    }

    /// Check the declared package variant without loading model weights.
    pub fn declared_variant<'a>(
        &self,
        manifest: &'a ModelManifest,
    ) -> Result<&'a ExecutionVariant> {
        self.validate()?;
        self.bind_variant(manifest, manifest.quantization)
    }

    /// Bind one model variant to every kernel plan selected for its tensor schemes.
    pub fn bind_mixed<'a>(
        &self,
        manifest: &'a ModelManifest,
        plans: &[ExecutionPlan],
    ) -> Result<&'a ExecutionVariant> {
        self.validate()?;
        if !plans.first().is_some_and(|first| {
            plans
                .iter()
                .any(|plan| plan.capability.quantization != first.capability.quantization)
        }) {
            return Err(RuntimeError::InvalidInput(
                "mixed precision requires kernel plans for multiple tensor schemes".to_owned(),
            ));
        }
        for plan in plans {
            self.check_plan(plan)?;
        }
        self.bind_variant(manifest, QuantizationScheme::Mixed)
    }

    fn check_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        if self.runtime_provider != plan.provider_name
            || self.kernel_provider != plan.kernel_provider_name
        {
            return Err(RuntimeError::InvalidInput(format!(
                "inference profile requires runtime {} and kernel {}, resolved {} and {}",
                self.runtime_provider,
                self.kernel_provider,
                plan.provider_name,
                plan.kernel_provider_name
            )));
        }
        if let Some(maximum) = plan.capability.maximum_sequence_length {
            if self.context_length > maximum {
                return Err(RuntimeError::InvalidInput(format!(
                    "inference profile context {} exceeds the kernel capacity {maximum}",
                    self.context_length
                )));
            }
        }
        Ok(())
    }

    fn bind_variant<'a>(
        &self,
        manifest: &'a ModelManifest,
        quantization: QuantizationScheme,
    ) -> Result<&'a ExecutionVariant> {
        let quantization_name = quantization.name();
        if self.quantization_variant != quantization_name || quantization != manifest.quantization {
            return Err(RuntimeError::InvalidInput(format!(
                "inference profile quantization {} does not match the resolved {quantization_name} model",
                self.quantization_variant
            )));
        }
        manifest
            .execution_variants
            .iter()
            .find(|variant| {
                variant.name == self.model_variant
                    && variant.quantization == quantization
                    && variant.runtime_provider == self.runtime_provider
                    && variant.kernel_provider == self.kernel_provider
            })
            .ok_or_else(|| {
                RuntimeError::InvalidInput(format!(
                    "model package {} has no matching execution variant {}",
                    manifest.model_id, self.model_variant
                ))
            })
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
    fn plans_logits_for_every_prefill_token() {
        let model = quantinfer_model::LlamaConfig::from_json(
            r#"{
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128
            }"#,
        )
        .unwrap()
        .to_model_spec("fixture")
        .unwrap();
        let plan = MemoryPlan::for_dense_prefill(&model, 4096, 16, 5).unwrap();
        assert_eq!(plan.weight_bytes, 4096);
        assert_eq!(plan.output_logits_bytes, 5 * 64 * 4);
        assert_eq!(plan.kv_cache_bytes, 2 * 16 * 2 * 4 * 2 * 4);
        assert_eq!(plan.temporary_workspace_bytes, 5 * 32 * 4 * 4 + 16 * 4);
        assert!(MemoryPlan::for_dense_prefill(&model, 4096, 16, 17).is_err());
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
