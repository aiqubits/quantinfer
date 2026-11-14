use std::collections::{btree_map::Entry, BTreeMap};

use quantinfer_core::{Shape, Tensor};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::TensorRole;

use crate::linear::{transpose_matrix, LinearWeight};
use crate::{Result, RuntimeError};

const ROLES: [TensorRole; 7] = [
    TensorRole::QueryProjection,
    TensorRole::KeyProjection,
    TensorRole::ValueProjection,
    TensorRole::OutputProjection,
    TensorRole::GateProjection,
    TensorRole::UpProjection,
    TensorRole::DownProjection,
];

/// An input-to-rank and rank-to-output pair for one dense projection.
#[derive(Debug, Clone)]
pub struct LoraProjection {
    down: Tensor,
    up: Tensor,
    scale: f32,
}

impl LoraProjection {
    pub fn new(down: Tensor, up: Tensor, scale: f32) -> Result<Self> {
        if !matches!(
            (down.shape().dims(), up.shape().dims()),
            ([_, rank], [up_rank, _]) if rank == up_rank && *rank != 0
        ) {
            return Err(invalid(
                "LoRA factors must have shapes [input, rank] and [rank, output]",
            ));
        }
        if !scale.is_finite()
            || down
                .as_f32()?
                .iter()
                .chain(up.as_f32()?)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("LoRA factors and scale must be finite"));
        }
        Ok(Self { down, up, scale })
    }

    /// Load factors from a PEFT linear projection in [rank, input] and [output, rank] order.
    pub fn from_peft(down: &Tensor, up: &Tensor, scale: f32) -> Result<Self> {
        if down.shape().rank() != 2 || up.shape().rank() != 2 {
            return Err(invalid("PEFT LoRA factors must be matrices"));
        }
        Self::new(transpose_matrix(down)?, transpose_matrix(up)?, scale)
    }

    fn validate_base(&self, shape: &Shape) -> Result<()> {
        let expected = [self.down.shape().dims()[0], self.up.shape().dims()[1]];
        if shape.dims() != expected {
            return Err(invalid(format!(
                "LoRA projection has shape {expected:?}, base weight has shape {:?}",
                shape.dims()
            )));
        }
        Ok(())
    }

    pub(crate) fn add_to(
        &self,
        backend: &impl KernelBackend,
        input: &Tensor,
        mut output: Tensor,
    ) -> Result<Tensor> {
        let low_rank = backend.matmul(input, &self.down)?;
        let delta = backend.matmul(&low_rank, &self.up)?;
        if output.shape() != delta.shape() {
            return Err(invalid("LoRA output does not match the base projection"));
        }
        for (value, &adjustment) in output.as_f32_mut()?.iter_mut().zip(delta.as_f32()?) {
            *value += self.scale * adjustment;
        }
        Ok(output)
    }
}

/// LoRA factors addressed by the semantic projection roles of a Dense model.
#[derive(Debug, Default)]
pub struct LoraAdapter {
    layers: BTreeMap<(usize, usize), LoraProjection>,
    head: Option<LoraProjection>,
}

impl LoraAdapter {
    pub fn insert(
        &mut self,
        layer: Option<usize>,
        role: TensorRole,
        projection: LoraProjection,
    ) -> Result<()> {
        if layer.is_none() && role == TensorRole::LmHead {
            if self.head.is_some() {
                return Err(invalid("duplicate LoRA LM head"));
            }
            self.head = Some(projection);
            return Ok(());
        }
        let layer = layer.ok_or_else(|| invalid("LoRA projection requires a layer index"))?;
        let index = ROLES
            .iter()
            .position(|candidate| *candidate == role)
            .ok_or_else(|| invalid(format!("unsupported LoRA projection {role:?}")))?;
        match self.layers.entry((layer, index)) {
            Entry::Occupied(_) => Err(invalid(format!("duplicate LoRA {role:?} at layer {layer}"))),
            Entry::Vacant(entry) => {
                entry.insert(projection);
                Ok(())
            }
        }
    }

    pub(crate) fn install(
        self,
        layers: &mut [[&mut LinearWeight; 7]],
        lm_head: &Tensor,
        head: &mut Option<LoraProjection>,
    ) -> Result<()> {
        if self.layers.is_empty() && self.head.is_none() {
            return Err(invalid("LoRA adapter has no projections"));
        }
        for (&(layer, index), projection) in &self.layers {
            let weights = layers
                .get(layer)
                .ok_or_else(|| invalid(format!("LoRA layer {layer} is outside the model")))?;
            let base = &weights[index];
            if matches!(base, LinearWeight::Lora { .. }) {
                return Err(invalid(format!(
                    "layer {layer} projection {:?} already has LoRA",
                    ROLES[index]
                )));
            }
            projection.validate_base(base.shape())?;
        }
        if let Some(projection) = &self.head {
            if head.is_some() {
                return Err(invalid("the LM head already has LoRA"));
            }
            projection.validate_base(lm_head.shape())?;
        }

        for ((layer, index), projection) in self.layers {
            let weight = &mut *layers[layer][index];
            let base = std::mem::replace(weight, LinearWeight::F32(Tensor::scalar(0.0)));
            *weight = LinearWeight::Lora {
                base: Box::new(base),
                projection,
            };
        }
        if let Some(projection) = self.head {
            *head = Some(projection);
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantinfer_backend_cpu::CpuScalar;
    use quantinfer_core::{DType, Q8_0Block, QuantizedTensor};

    #[test]
    fn adds_a_low_rank_delta_to_a_packed_base_projection() {
        let base = LinearWeight::Q8_0(
            QuantizedTensor::from_blocks(
                vec![32, 2],
                vec![
                    Q8_0Block {
                        scale: 0.5,
                        values: [1; 32]
                    };
                    2
                ],
            )
            .unwrap(),
        );
        let input = Tensor::from_f32(vec![2, 32], vec![1.0; 64]).unwrap();
        let down = Tensor::from_f32(vec![1, 32], vec![0.5; 32]).unwrap();
        let up = Tensor::from_f32(vec![2, 1], vec![2.0, -3.0]).unwrap();
        let projection = LoraProjection::from_peft(&down, &up, 0.25).unwrap();
        let weight = LinearWeight::Lora {
            base: Box::new(base),
            projection,
        };
        assert_eq!(weight.dtype(), DType::Q8_0);
        assert_eq!(weight.shape().dims(), [32, 2]);
        let output = weight.matmul(&CpuScalar::new(), &input).unwrap();
        assert_eq!(output.as_f32().unwrap(), [24.0, 4.0, 24.0, 4.0]);
    }

    #[test]
    fn validates_all_adapter_targets_before_changing_weights() {
        let mut weights = std::array::from_fn::<_, 7, _>(|_| {
            LinearWeight::F32(Tensor::zeros_f32(vec![2, 2]).unwrap())
        });
        let make_projection = |input, output| {
            LoraProjection::new(
                Tensor::zeros_f32(vec![input, 1]).unwrap(),
                Tensor::zeros_f32(vec![1, output]).unwrap(),
                1.0,
            )
            .unwrap()
        };
        let mut adapter = LoraAdapter::default();
        adapter
            .insert(Some(0), TensorRole::QueryProjection, make_projection(2, 2))
            .unwrap();
        adapter
            .insert(Some(0), TensorRole::DownProjection, make_projection(4, 2))
            .unwrap();
        let mut head = None;
        let lm_head = Tensor::zeros_f32(vec![2, 4]).unwrap();
        let result = adapter.install(&mut [weights.each_mut()], &lm_head, &mut head);
        assert!(result.is_err());
        assert!(weights
            .iter()
            .all(|weight| matches!(weight, LinearWeight::F32(_))));
        assert!(head.is_none());
    }
}
