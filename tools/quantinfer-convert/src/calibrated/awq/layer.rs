use crate::calibrated::layer::Layer;
use quantinfer::{kernel::KernelBackend, model::TensorRole};
use quantinfer_adapter_awq::{AwqCalibration, AwqProvider, AwqScalePlan};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use quantinfer_quant::QuantizationError;
use serde_json::{json, Value};
use std::mem::size_of;
use TensorRole::*;

#[derive(Clone, Copy)]
enum Block {
    Attention,
    FeedForward,
    Linear,
}

impl Layer<'_> {
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
        provider: &AwqProvider,
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
        let result = provider.search_scales(&weights, &statistics, |candidate| {
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
        provider: &AwqProvider,
        roles: &[TensorRole],
        upstream: TensorRole,
        input: &[Tensor],
        block: Block,
    ) -> Result<Value, String> {
        let plan = self.search(provider, roles, input, block)?;
        self.fold(roles, upstream, &plan)?;
        Ok(
            json!({"layer":self.index,"roles":roles,"upstream":upstream,"ratio":plan.ratio,
            "loss":plan.loss,"candidate_losses":plan.candidate_losses}),
        )
    }

    pub fn scale(
        &mut self,
        provider: &AwqProvider,
        hidden: &[Tensor],
    ) -> Result<Vec<Value>, String> {
        let mut records = Vec::with_capacity(4);
        let attention = self.normalized(InputNormalization, hidden)?;
        records.push(self.search_and_fold(
            provider,
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
            let observed = self.observe(&attention, OutputProjection)?;
            records.push(self.search_and_fold(
                provider,
                &[OutputProjection],
                ValueProjection,
                &observed,
                Block::Linear,
            )?);
        }
        let residual = self.ffn_residual(hidden, &attention)?;
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        records.push(self.search_and_fold(
            provider,
            &[GateProjection, UpProjection],
            PostAttentionNormalization,
            &feed_forward,
            Block::FeedForward,
        )?);
        drop(feed_forward);
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        let down = self.observe(&feed_forward, DownProjection)?;
        records.push(self.search_and_fold(
            provider,
            &[DownProjection],
            UpProjection,
            &down,
            Block::Linear,
        )?);
        Ok(records)
    }

    pub fn clip(
        &mut self,
        provider: &AwqProvider,
        hidden: &[Tensor],
    ) -> Result<Vec<Value>, String> {
        let attention = self.normalized(InputNormalization, hidden)?;
        let output = self.observe(&attention, OutputProjection)?;
        let residual = self.ffn_residual(hidden, &attention)?;
        let feed_forward = self.normalized(PostAttentionNormalization, &residual)?;
        let down = self.observe(&feed_forward, DownProjection)?;
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
                let plan = provider
                    .search_clipping(self.weight(role)?, &input)
                    .map_err(|e| format!("AWQ layer {} {role:?} clipping: {e}", self.index))?;
                records.push(json!({"layer":self.index,"role":role,"rows":plan.calibration_rows,
                    "group_loss_before":plan.group_loss_before,"group_loss_after":plan.group_loss_after}));
                self.set_weight(role, plan.weights)?;
            }
        }
        Ok(records)
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
