use super::{invalid, matrix_shape};
use quantinfer_core::Tensor;
use quantinfer_quant::Result;

pub(super) fn validate_scales(scales: &[f32], width: usize) -> Result<()> {
    if scales.len() != width || scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
        return Err(invalid(
            "AWQ channel scales must be finite, positive and match the projection",
        ));
    }
    Ok(())
}
fn map_values(weight: &Tensor, mut apply: impl FnMut(usize, f32) -> f32) -> Result<Tensor> {
    let mut values = Vec::new();
    let source = weight.as_f32()?;
    values
        .try_reserve_exact(source.len())
        .map_err(|_| invalid("cannot allocate AWQ transform"))?;
    for (i, &value) in source.iter().enumerate() {
        let value = apply(i, value);
        if !value.is_finite() {
            return Err(invalid("AWQ equivalent transform exceeds finite F32"));
        }
        values.push(value);
    }
    Ok(Tensor::from_f32(weight.shape().clone(), values)?)
}
pub(super) fn normalization(weight: &Tensor, scales: &[f32]) -> Result<Tensor> {
    if weight.shape().dims() != [scales.len()] {
        return Err(invalid("AWQ normalization must be a channel vector"));
    }
    validate_scales(scales, scales.len())?;
    map_values(weight, |i, v| v / scales[i])
}
pub(super) fn input_weight(weight: &Tensor, scales: &[f32]) -> Result<Tensor> {
    let (k, n) = matrix_shape(weight.shape())?;
    validate_scales(scales, k)?;
    map_values(weight, |i, v| v * scales[i / n])
}
pub(super) fn undo_input_weight(weight: &Tensor, scales: &[f32]) -> Result<Tensor> {
    let (k, n) = matrix_shape(weight.shape())?;
    validate_scales(scales, k)?;
    map_values(weight, |i, v| v / scales[i / n])
}
pub(super) fn output_weight(weight: &Tensor, scales: &[f32]) -> Result<Tensor> {
    let (_, n) = matrix_shape(weight.shape())?;
    validate_scales(scales, n)?;
    map_values(weight, |i, v| v / scales[i % n])
}
