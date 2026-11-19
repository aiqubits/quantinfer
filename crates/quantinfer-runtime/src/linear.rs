//! Shared dense linear-weight loading and tensor transforms.

use quantinfer_core::{
    DType, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K, QuantizedTensorQ5K,
    QuantizedTensorQ6K, Shape, Tensor,
};
use quantinfer_format::{GgmlType, GgufReader};
use quantinfer_kernel::KernelBackend;

use crate::{LoraProjection, Result, RuntimeError};

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum LinearWeight {
    F32(Tensor),
    Q8_0(QuantizedTensor),
    Q4_0(QuantizedTensor4),
    Q4_K(QuantizedTensorQ4K),
    Q5_K(QuantizedTensorQ5K),
    Q6_K(QuantizedTensorQ6K),
    Q4_KOutputRows(QuantizedTensorQ4K),
    Q5_KOutputRows(QuantizedTensorQ5K),
    Q6_KOutputRows(QuantizedTensorQ6K),
    Q8_0OutputRows(QuantizedTensor),
    Q4_0OutputRows(QuantizedTensor4),
    Lora {
        base: Box<LinearWeight>,
        projection: LoraProjection,
    },
}

impl LinearWeight {
    pub fn shape(&self) -> &Shape {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Q8_0(tensor) | Self::Q8_0OutputRows(tensor) => tensor.shape(),
            Self::Q4_0(tensor) | Self::Q4_0OutputRows(tensor) => tensor.shape(),
            Self::Q4_K(tensor) | Self::Q4_KOutputRows(tensor) => tensor.shape(),
            Self::Q5_K(tensor) | Self::Q5_KOutputRows(tensor) => tensor.shape(),
            Self::Q6_K(tensor) | Self::Q6_KOutputRows(tensor) => tensor.shape(),
            Self::Lora { base, .. } => base.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::Q8_0(_) | Self::Q8_0OutputRows(_) => DType::Q8_0,
            Self::Q4_0(_) | Self::Q4_0OutputRows(_) => DType::Q4_0,
            Self::Q4_K(_) | Self::Q4_KOutputRows(_) => DType::Q4_K,
            Self::Q5_K(_) | Self::Q5_KOutputRows(_) => DType::Q5_K,
            Self::Q6_K(_) | Self::Q6_KOutputRows(_) => DType::Q6_K,
            Self::Lora { base, .. } => base.dtype(),
        }
    }

    pub(crate) fn matmul<B: KernelBackend>(&self, backend: &B, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::F32(weight) => Ok(backend.matmul(input, weight)?),
            Self::Q8_0(weight) => Ok(backend.matmul_q8_0(input, weight)?),
            Self::Q4_0(weight) => Ok(backend.matmul_q4_0(input, weight)?),
            Self::Q4_K(weight) => Ok(backend.matmul_q4_k(input, weight)?),
            Self::Q5_K(weight) => Ok(backend.matmul_q5_k(input, weight)?),
            Self::Q6_K(weight) => Ok(backend.matmul_q6_k(input, weight)?),
            Self::Q4_KOutputRows(weight) => Ok(backend.matmul_q4_k_output_rows(input, weight)?),
            Self::Q5_KOutputRows(weight) => Ok(backend.matmul_q5_k_output_rows(input, weight)?),
            Self::Q6_KOutputRows(weight) => Ok(backend.matmul_q6_k_output_rows(input, weight)?),
            Self::Q8_0OutputRows(weight) => Ok(backend.matmul_q8_0_output_rows(input, weight)?),
            Self::Q4_0OutputRows(weight) => Ok(backend.matmul_q4_0_output_rows(input, weight)?),
            Self::Lora { base, projection } => {
                projection.add_to(backend, input, base.matmul(backend, input)?)
            }
        }
    }
}

pub(crate) fn load_linear(reader: &GgufReader, name: &str) -> Result<LinearWeight> {
    if reader.matrix_uses_output_rows(name)? {
        return crate::ggml::load_ggml_linear(reader, name, None);
    }
    match reader.tensor_info(name)?.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            Ok(LinearWeight::F32(reader.tensor_f32(name)?))
        }
        GgmlType::Q8_0 => Ok(LinearWeight::Q8_0(reader.tensor_q8_0(name)?)),
        GgmlType::Q4_0 => Ok(LinearWeight::Q4_0(reader.tensor_q4_0(name)?)),
        GgmlType::Q4_K => Ok(LinearWeight::Q4_K(reader.tensor_q4_k(name)?)),
        GgmlType::Q6_K => Ok(LinearWeight::Q6_K(reader.tensor_q6_k(name)?)),
        GgmlType::Q5_K => Ok(LinearWeight::Q5_K(reader.tensor_q5_k(name)?)),
    }
}

/// Decode a floating native-view tensor, honoring a matrix's physical layout.
/// GGML embedding/table reshaping remains in the dimension-order adapter.
pub(crate) fn load_native_float(reader: &GgufReader, name: &str) -> Result<Tensor> {
    let info = reader.tensor_info(name)?;
    if !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        return Err(RuntimeError::InvalidInput(format!(
            "native model tensor {name} requires floating point storage"
        )));
    }
    if info.shape.len() == 2 && reader.matrix_uses_output_rows(name)? {
        crate::ggml::load_ggml_matrix(reader, name, None)
    } else {
        reader.tensor_storage_layout(name)?;
        Ok(reader.tensor_f32(name)?)
    }
}

pub(crate) fn transpose_matrix(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(RuntimeError::InvalidInput(
            "tensor transform requires a matrix".to_owned(),
        ));
    }
    let rows = dims[0];
    let columns = dims[1];
    let source = tensor.as_f32()?;
    let mut values = vec![0.0; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            values[column * rows + row] = source[row * columns + column];
        }
    }
    Ok(Tensor::from_f32(vec![columns, rows], values)?)
}

pub(crate) fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape() != rhs.shape() {
        return Err(RuntimeError::InvalidInput(
            "residual tensors must have identical shapes".to_owned(),
        ));
    }
    let values = lhs
        .as_f32()?
        .iter()
        .zip(rhs.as_f32()?)
        .map(|(left, right)| left + right)
        .collect();
    Ok(Tensor::from_f32(lhs.shape().clone(), values)?)
}

#[cfg(test)]
mod tests;
