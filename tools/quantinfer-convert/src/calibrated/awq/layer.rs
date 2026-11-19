use crate::calibrated::model;
use quantinfer::{
    cpu::CpuModel,
    kernel::KernelBackend,
    model::{ModelSpec, TensorRole},
    runtime::{LinearWeight, ProjectionObserver, RuntimeError},
};
use quantinfer_adapter_awq::{AwqCalibration, AwqProvider, AwqScalePlan};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use quantinfer_quant::QuantizationError;
use serde_json::{json, Value};
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
        .ok_or("AWQ row count overflows")?;
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
        .ok_or("AWQ observation width overflows")?;
    let required = width
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| n.checked_add(prompts.len().checked_mul(32 * size_of::<Tensor>())?))
        .ok_or("AWQ activation workspace overflows")?;
    if required > budget {
        return Err(format!(
            "AWQ activation workspace needs {required} bytes, above the statistics budget {budget}"
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
    pub provider: &'a AwqProvider,
}

#[derive(Clone, Copy)]
enum Block {
    Attention,
    FeedForward,
    Linear,
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

    fn set_weight(&mut self, role: TensorRole, weight: Tensor) -> Result<(), String> {
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

    fn normalized(&mut self, role: TensorRole, input: &[Tensor]) -> Result<Vec<Tensor>, String> {
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

    fn attention(
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

    fn ffn(
        &self,
        input: &Tensor,
        observer: Option<&mut dyn ProjectionObserver>,
    ) -> Result<Tensor, String> {
        self.model
            .forward_feed_forward_block(self.index, input, observer)
            .map_err(|e| e.to_string())
    }

    fn ffn_residual(
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
    fn observe(
        &self,
        input: &[Tensor],
        block: Block,
        role: TensorRole,
    ) -> Result<Vec<Tensor>, String> {
        input
            .iter()
            .map(|x| {
                let mut observation = None;
                let mut observer = |_: usize, roles: &[TensorRole], values: &Tensor| {
                    if roles.contains(&role) {
                        if observation.is_some() {
                            return Err(RuntimeError::InvalidInput(format!(
                                "duplicate AWQ observation for {role:?}"
                            )));
                        }
                        observation = Some(values.clone());
                    }
                    Ok(())
                };
                match block {
                    Block::Attention => self.attention(x, Some(&mut observer))?,
                    Block::FeedForward => self.ffn(x, Some(&mut observer))?,
                    Block::Linear => return Err("AWQ observer requires a decoder block".into()),
                };
                observation.ok_or_else(|| format!("layer {} did not observe {role:?}", self.index))
            })
            .collect()
    }

    fn evaluate(&self, input: &Tensor, block: Block, linear: &Tensor) -> Result<Tensor, String> {
        match block {
            Block::Attention => self.attention(input, None),
            Block::FeedForward => self.ffn(input, None),
            Block::Linear => CpuScalar::new()
                .matmul(input, linear)
                .map_err(|e| e.to_string()),
        }
    }

    fn search(
        &mut self,
        roles: &[TensorRole],
        input: &[Tensor],
        block: Block,
    ) -> Result<AwqScalePlan, String> {
        let originals: Vec<_> = roles
            .iter()
            .map(|&r| self.weight(r).cloned())
            .collect::<Result<_, _>>()?;
        let weights: Vec<_> = originals.iter().collect();
        let width = originals[0].shape().dims()[0];
        let bytes = width
            .checked_mul(size_of::<f64>())
            .ok_or("AWQ statistics overflow")?;
        let mut statistics = AwqCalibration::new(width, bytes).map_err(|e| e.to_string())?;
        for x in input {
            statistics.observe_rows(x).map_err(|e| e.to_string())?;
        }
        let reference: Vec<_> = input
            .iter()
            .map(|x| self.evaluate(x, block, &originals[0]))
            .collect::<Result<_, _>>()?;
        let result = self
            .provider
            .search_scales(&weights, &statistics, |candidate| {
                let evaluate = || -> Result<f64, String> {
                    let linear = if matches!(block, Block::Linear) {
                        candidate
                            .into_iter()
                            .next()
                            .ok_or("AWQ candidate has no projection")?
                    } else {
                        for (&role, weight) in roles.iter().zip(candidate) {
                            self.set_weight(role, weight)?;
                        }
                        // The block path reads the temporary weights from the model.
                        return self.mean_loss(input, &reference, block, &originals[0]);
                    };
                    self.mean_loss(input, &reference, block, &linear)
                };
                evaluate().map_err(QuantizationError::InvalidInput)
            });
        if !matches!(block, Block::Linear) {
            for (&role, original) in roles.iter().zip(originals) {
                self.set_weight(role, original)?;
            }
        }
        result.map_err(|e| format!("AWQ layer {} {roles:?} scale search: {e}", self.index))
    }

    fn mean_loss(
        &self,
        input: &[Tensor],
        reference: &[Tensor],
        block: Block,
        linear: &Tensor,
    ) -> Result<f64, String> {
        let mut squared = 0.0;
        let mut count = 0usize;
        for (x, expected) in input.iter().zip(reference) {
            let actual = self.evaluate(x, block, linear)?;
            if actual.shape() != expected.shape() {
                return Err("AWQ block output shape changed".into());
            }
            let values = actual.as_f32().map_err(|e| e.to_string())?;
            let expected = expected.as_f32().map_err(|e| e.to_string())?;
            count = count
                .checked_add(values.len())
                .ok_or("AWQ loss count overflows")?;
            squared += values
                .iter()
                .zip(expected)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>();
        }
        Ok(squared / count as f64)
    }

    fn fold(
        &mut self,
        roles: &[TensorRole],
        upstream: TensorRole,
        plan: &AwqScalePlan,
    ) -> Result<(), String> {
        match upstream {
            InputNormalization | PostAttentionNormalization => {
                let norm = self.norm(upstream)?;
                *norm = plan.normalization(norm).map_err(|e| e.to_string())?;
            }
            UpProjection | ValueProjection => {
                let adjusted = plan
                    .output_weight(self.weight(upstream)?)
                    .map_err(|e| e.to_string())?;
                self.set_weight(upstream, adjusted)?;
            }
            _ => return Err(format!("invalid AWQ upstream role {upstream:?}")),
        }
        for &role in roles {
            let adjusted = plan
                .input_weight(self.weight(role)?)
                .map_err(|e| e.to_string())?;
            self.set_weight(role, adjusted)?;
        }
        Ok(())
    }

    fn search_and_fold(
        &mut self,
        roles: &[TensorRole],
        upstream: TensorRole,
        input: &[Tensor],
        block: Block,
    ) -> Result<Value, String> {
        let plan = self.search(roles, input, block)?;
        self.fold(roles, upstream, &plan)?;
        Ok(
            json!({"layer":self.index,"roles":roles,"upstream":upstream,"ratio":plan.ratio,
            "loss":plan.loss,"candidate_losses":plan.candidate_losses}),
        )
    }

    pub fn scale(&mut self, hidden: &[Tensor]) -> Result<Vec<Value>, String> {
        let mut records = Vec::with_capacity(4);
        let attention = self.normalized(InputNormalization, hidden)?;
        records.push(self.search_and_fold(
            &[QueryProjection, KeyProjection, ValueProjection],
            InputNormalization,
            &attention,
            Block::Attention,
        )?);
        drop(attention);
        let attention = self.normalized(InputNormalization, hidden)?;
        // GQA repeats value heads. The one-to-one V/O transform applies when
        // those two widths match.
        if self.weight(ValueProjection)?.shape().dims()[1]
            == self.weight(OutputProjection)?.shape().dims()[0]
        {
            let observed = self.observe(&attention, Block::Attention, OutputProjection)?;
            records.push(self.search_and_fold(
                &[OutputProjection],
                ValueProjection,
                &observed,
                Block::Linear,
            )?);
        }
        let residual = self.ffn_residual(hidden, &attention)?;
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        records.push(self.search_and_fold(
            &[GateProjection, UpProjection],
            PostAttentionNormalization,
            &feed_forward,
            Block::FeedForward,
        )?);
        drop(feed_forward);
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        let down = self.observe(&feed_forward, Block::FeedForward, DownProjection)?;
        records.push(self.search_and_fold(
            &[DownProjection],
            UpProjection,
            &down,
            Block::Linear,
        )?);
        Ok(records)
    }

    pub fn clip(&mut self, hidden: &[Tensor]) -> Result<Vec<Value>, String> {
        let attention = self.normalized(InputNormalization, hidden)?;
        let output = self.observe(&attention, Block::Attention, OutputProjection)?;
        let residual = self.ffn_residual(hidden, &attention)?;
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        let down = self.observe(&feed_forward, Block::FeedForward, DownProjection)?;
        // All observations above use the same transformed floating layer.
        // Clip projections with linear reconstruction objectives, excluding Q/K.
        let mut records = Vec::with_capacity(5);
        for (roles, samples) in [
            (&[ValueProjection][..], attention),
            (&[OutputProjection], output),
            (&[GateProjection, UpProjection], feed_forward),
            (&[DownProjection], down),
        ] {
            let input = concatenate(&samples)?;
            drop(samples);
            for &role in roles {
                let plan = self
                    .provider
                    .search_clipping(self.weight(role)?, &input)
                    .map_err(|e| format!("AWQ layer {} {role:?} clipping: {e}", self.index))?;
                records.push(json!({"layer":self.index,"role":role,"rows":plan.calibration_rows,
                    "group_loss_before":plan.group_loss_before,"group_loss_after":plan.group_loss_after}));
                self.set_weight(role, plan.weights)?;
            }
        }
        Ok(records)
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

fn concatenate(input: &[Tensor]) -> Result<Tensor, String> {
    let width = input
        .first()
        .ok_or("AWQ has no observations")?
        .shape()
        .dims()[1];
    let rows = input
        .iter()
        .try_fold(0usize, |sum, t| sum.checked_add(t.shape().dims()[0]))
        .ok_or("AWQ clipping row count overflows")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            rows.checked_mul(width)
                .ok_or("AWQ clipping shape overflows")?,
        )
        .map_err(|e| e.to_string())?;
    for tensor in input {
        if tensor.shape().dims()[1] != width {
            return Err("AWQ clipping observation width changed".into());
        }
        values.extend_from_slice(tensor.as_f32().map_err(|e| e.to_string())?);
    }
    Tensor::from_f32(vec![rows, width], values).map_err(|e| e.to_string())
}

fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape() != b.shape() {
        return Err("AWQ residual shape disagrees with block output".into());
    }
    let a_values = a.as_f32().map_err(|e| e.to_string())?;
    let b_values = b.as_f32().map_err(|e| e.to_string())?;
    let mut sum = Vec::new();
    sum.try_reserve_exact(a_values.len())
        .map_err(|e| e.to_string())?;
    sum.extend(a_values.iter().zip(b_values).map(|(a, b)| a + b));
    Tensor::from_f32(a.shape().clone(), sum).map_err(|e| e.to_string())
}
