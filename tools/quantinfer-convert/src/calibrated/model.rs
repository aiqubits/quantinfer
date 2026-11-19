//! Semantic projection access at the offline workflow composition boundary.

use quantinfer::{
    cpu::{CpuModel, DenseModel},
    model::TensorRole,
    runtime::LinearWeight,
};

pub(super) const GROUPS: [&[TensorRole]; 4] = [
    &[
        TensorRole::QueryProjection,
        TensorRole::KeyProjection,
        TensorRole::ValueProjection,
    ],
    &[TensorRole::OutputProjection],
    &[TensorRole::GateProjection, TensorRole::UpProjection],
    &[TensorRole::DownProjection],
];

pub(super) fn projection(
    model: &CpuModel,
    layer: usize,
    role: TensorRole,
) -> Result<&LinearWeight, String> {
    macro_rules! select {
        ($weights:expr) => {
            match role {
                TensorRole::QueryProjection => &$weights.q_proj,
                TensorRole::KeyProjection => &$weights.k_proj,
                TensorRole::ValueProjection => &$weights.v_proj,
                TensorRole::OutputProjection => &$weights.o_proj,
                TensorRole::GateProjection => &$weights.gate_proj,
                TensorRole::UpProjection => &$weights.up_proj,
                TensorRole::DownProjection => &$weights.down_proj,
                _ => return Err(format!("{role:?} is not a decoder linear projection")),
            }
        };
    }
    Ok(match model {
        DenseModel::Qwen3(model) => {
            let weights = model
                .layers
                .get(layer)
                .ok_or("layer index is out of range")?;
            select!(weights)
        }
        DenseModel::Llama(model) => {
            let weights = model
                .layers
                .get(layer)
                .ok_or("layer index is out of range")?;
            select!(weights)
        }
    })
}

pub(super) fn replace_projection(
    model: &mut CpuModel,
    layer: usize,
    role: TensorRole,
    replacement: LinearWeight,
) -> Result<(), String> {
    if replacement.shape() != projection(model, layer, role)?.shape() {
        return Err(format!(
            "replacement {role:?} shape disagrees with the model"
        ));
    }
    macro_rules! select {
        ($weights:expr) => {
            match role {
                TensorRole::QueryProjection => &mut $weights.q_proj,
                TensorRole::KeyProjection => &mut $weights.k_proj,
                TensorRole::ValueProjection => &mut $weights.v_proj,
                TensorRole::OutputProjection => &mut $weights.o_proj,
                TensorRole::GateProjection => &mut $weights.gate_proj,
                TensorRole::UpProjection => &mut $weights.up_proj,
                TensorRole::DownProjection => &mut $weights.down_proj,
                _ => return Err(format!("{role:?} is not a decoder linear projection")),
            }
        };
    }
    let target = match model {
        DenseModel::Qwen3(model) => {
            let weights = model
                .layers
                .get_mut(layer)
                .ok_or("layer index is out of range")?;
            select!(weights)
        }
        DenseModel::Llama(model) => {
            let weights = model
                .layers
                .get_mut(layer)
                .ok_or("layer index is out of range")?;
            select!(weights)
        }
    };
    *target = replacement;
    Ok(())
}
