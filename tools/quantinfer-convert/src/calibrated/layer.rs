//! Shared dense layer execution for sequential offline calibration.
use super::model;
use quantinfer::{
    cpu::CpuModel,
    kernel::KernelBackend,
    model::{ModelSpec, TensorRole},
    runtime::{LinearWeight, ProjectionObserver, RuntimeError},
};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use std::mem::size_of;
use TensorRole::*;

pub(super) fn embed_prompts(
    model: &CpuModel,
    spec: &ModelSpec,
    prompts: &[Vec<usize>],
    budget: usize,
) -> Result<Vec<Tensor>, String> {
    let rows = prompts
        .iter()
        .try_fold(0usize, |sum, p| sum.checked_add(p.len()))
        .ok_or("calibration row count overflows")?;
    let attention = &spec.layers[0].attention;
    // Conservative bound for corpus activations, reference outputs, clipping
    // flattening, and one layer's evaluation/KV intermediates.
    let width = spec
        .hidden_size
        .checked_mul(16)
        .and_then(|n| n.checked_add(spec.intermediate_size.checked_mul(8)?))
        .and_then(|n| {
            n.checked_add(
                attention
                    .num_heads
                    .checked_mul(attention.head_dim)?
                    .checked_mul(8)?,
            )
        })
        .ok_or("calibration observation width overflows")?;
    let required = width
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| n.checked_add(prompts.len().checked_mul(32 * size_of::<Tensor>())?))
        .ok_or("calibration activation workspace overflows")?;
    if required > budget {
        return Err(format!(
            "calibration activation workspace needs {required} bytes, above the statistics budget {budget}"
        ));
    }
    let embedding = match model {
        CpuModel::Qwen3(model) => &model.embedding,
        CpuModel::Llama(model) => &model.embedding,
    };
    prompts
        .iter()
        .map(|tokens| {
            CpuScalar::new()
                .embedding(embedding, tokens)
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub(super) struct Layer<'a> {
    pub model: &'a mut CpuModel,
    pub index: usize,
    pub context: usize,
}

impl Layer<'_> {
    pub fn weight(&self, role: TensorRole) -> Result<&Tensor, String> {
        match model::projection(self.model, self.index, role)? {
            LinearWeight::F32(w) => Ok(w),
            _ => Err(format!(
                "layer {} {role:?} must be floating point",
                self.index
            )),
        }
    }

    #[cfg(feature = "awq")]
    pub fn set_weight(&mut self, role: TensorRole, weight: Tensor) -> Result<(), String> {
        model::replace_projection(self.model, self.index, role, LinearWeight::F32(weight))
    }

    pub fn norm(&mut self, role: TensorRole) -> Result<&mut Tensor, String> {
        macro_rules! get {
            ($model:expr) => {{
                let layer = &mut $model.layers[self.index];
                match role {
                    InputNormalization => &mut layer.input_layernorm,
                    PostAttentionNormalization => &mut layer.post_attention_layernorm,
                    _ => return Err(format!("{role:?} is not a foldable normalization")),
                }
            }};
        }
        Ok(match self.model {
            CpuModel::Qwen3(model) => get!(model),
            CpuModel::Llama(model) => get!(model),
        })
    }

    pub fn normalized(
        &mut self,
        role: TensorRole,
        input: &[Tensor],
    ) -> Result<Vec<Tensor>, String> {
        let epsilon = match &self.model {
            CpuModel::Qwen3(model) => model.config.rms_norm_eps,
            CpuModel::Llama(model) => model.config.rms_norm_eps,
        };
        let weight = self.norm(role)?;
        input
            .iter()
            .map(|x| {
                CpuScalar::new()
                    .rms_norm(x, weight, epsilon)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    pub fn attention(
        &self,
        input: &Tensor,
        observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor, String> {
        let mut cache = self
            .model
            .new_cache(self.context)
            .map_err(|e| e.to_string())?;
        self.model
            .forward_attention_block(self.index, input, &mut cache, observer)
            .map_err(|e| e.to_string())
    }

    pub fn ffn(
        &self,
        input: &Tensor,
        observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor, String> {
        self.model
            .forward_feed_forward_block(self.index, input, observer)
            .map_err(|e| e.to_string())
    }

    pub fn ffn_residual(
        &self,
        hidden: &[Tensor],
        normalized: &[Tensor],
    ) -> Result<Vec<Tensor>, String> {
        hidden
            .iter()
            .zip(normalized)
            .map(|(x, y)| add(x, &self.attention(y, None)?))
            .collect()
    }

    // Preserve prompt boundaries for attention. Linear searches and clipping
    // may concatenate rows because they do not share sequence state.
    pub fn observe(&self, input: &[Tensor], role: TensorRole) -> Result<Vec<Tensor>, String> {
        input
            .iter()
            .map(|x| {
                let mut observation = None;
                let mut observer = |_: usize, roles: &[TensorRole], values: &Tensor| {
                    if roles.contains(&role) {
                        if observation.is_some() {
                            return Err(RuntimeError::InvalidInput(format!(
                                "duplicate calibration observation for {role:?}"
                            )));
                        }
                        observation = Some(values.clone());
                    }
                    Ok(())
                };
                match role {
                    OutputProjection => self.attention(x, Some(&mut observer))?,
                    DownProjection => self.ffn(x, Some(&mut observer))?,
                    _ => return Err(format!("no block-specific input observer for {role:?}")),
                };
                observation.ok_or_else(|| format!("layer {} did not observe {role:?}", self.index))
            })
            .collect()
    }

    pub fn advance(&mut self, hidden: &mut [Tensor]) -> Result<(), String> {
        let attention = self.normalized(InputNormalization, hidden)?;
        let residual = self.ffn_residual(hidden, &attention)?;
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        for ((next, x), input) in hidden.iter_mut().zip(residual).zip(feed_forward) {
            *next = add(&x, &self.ffn(&input, None)?)?;
        }
        Ok(())
    }
}

fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape() != b.shape() {
        return Err("calibration residual shape disagrees with block output".into());
    }
    let a_values = a.as_f32().map_err(|e| e.to_string())?;
    let b_values = b.as_f32().map_err(|e| e.to_string())?;
    let mut sum = Vec::new();
    sum.try_reserve_exact(a_values.len())
        .map_err(|e| e.to_string())?;
    sum.extend(a_values.iter().zip(b_values).map(|(a, b)| a + b));
    Tensor::from_f32(a.shape().clone(), sum).map_err(|e| e.to_string())
}
