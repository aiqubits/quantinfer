//! Shared GGML output-row decoding and layout transforms.

use quantinfer_core::{
    QuantizedTensor, QuantizedTensor4, Tensor, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_format::{qk_channel, GgmlType, GgufReader};
use quantinfer_quant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};

use crate::{linear::LinearWeight, Result, RuntimeError};

pub(crate) fn load_ggml_as_float(reader: &GgufReader, name: &str) -> Result<Tensor> {
    match reader.tensor_info(name)?.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => return Ok(reader.tensor_f32(name)?),
        GgmlType::Q8_0 => dequantize_q8_0(&reader.tensor_q8_0(name)?),
        GgmlType::Q4_0 => dequantize_q4_0(&reader.tensor_q4_0(name)?),
        GgmlType::Q4_K => dequantize_q4_k(&reader.tensor_q4_k(name)?),
        GgmlType::Q5_K => dequantize_q5_k(&reader.tensor_q5_k(name)?),
        GgmlType::Q6_K => dequantize_q6_k(&reader.tensor_q6_k(name)?),
    }
    .map_err(|error| {
        RuntimeError::InvalidInput(format!("cannot decode GGML tensor {name}: {error}"))
    })
}

pub(crate) fn load_ggml_linear(
    reader: &GgufReader,
    name: &str,
    rope_heads: Option<usize>,
) -> Result<LinearWeight> {
    let info = reader.tensor_info(name)?;
    let &[input, output] = info.shape.as_slice() else {
        return Err(RuntimeError::InvalidInput(format!(
            "GGML matrix {name} must be rank 2"
        )));
    };
    match info.dtype {
        GgmlType::Q8_0 if input != 0 && input.is_multiple_of(Q8_0_BLOCK_SIZE) => {
            let weight = reader.tensor_q8_0(name)?;
            let weight = if let Some(dim) = rope_head_dim(name, output, rope_heads)? {
                QuantizedTensor::from_blocks(
                    weight.shape().clone(),
                    reorder_ggml_rows(weight.blocks(), input / Q8_0_BLOCK_SIZE, dim),
                )?
            } else {
                weight
            };
            Ok(LinearWeight::Q8_0OutputRows(weight))
        }
        GgmlType::Q4_0 if input != 0 && input.is_multiple_of(Q4_0_BLOCK_SIZE) => {
            let weight = reader.tensor_q4_0(name)?;
            let weight = if let Some(dim) = rope_head_dim(name, output, rope_heads)? {
                QuantizedTensor4::from_blocks(
                    weight.shape().clone(),
                    reorder_ggml_rows(weight.blocks(), input / Q4_0_BLOCK_SIZE, dim),
                )?
            } else {
                weight
            };
            Ok(LinearWeight::Q4_0OutputRows(weight))
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Ok(LinearWeight::F32(load_ggml_matrix(
            reader, name, rope_heads,
        )?)),
        dtype => Err(RuntimeError::InvalidInput(format!(
            "unsupported GGML projection {name} with input width {input} and type {dtype:?}"
        ))),
    }
}

fn rope_head_dim(name: &str, output: usize, rope_heads: Option<usize>) -> Result<Option<usize>> {
    rope_heads
        .map(|heads| {
            if heads == 0
                || output == 0
                || !output.is_multiple_of(heads)
                || !(output / heads).is_multiple_of(2)
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "GGML matrix {name} has an invalid RoPE layout"
                )));
            }
            Ok(output / heads)
        })
        .transpose()
}

fn reorder_ggml_rows<T: Clone>(blocks: &[T], row_blocks: usize, dim: usize) -> Vec<T> {
    let mut ordered = blocks.to_vec();
    for (row, values) in blocks.chunks_exact(row_blocks).enumerate() {
        let start = qk_channel(row, dim) * row_blocks;
        ordered[start..start + row_blocks].clone_from_slice(values);
    }
    ordered
}

pub(crate) fn load_ggml_matrix(
    reader: &GgufReader,
    name: &str,
    rope_heads: Option<usize>,
) -> Result<Tensor> {
    let tensor = load_ggml_as_float(reader, name)?;
    let &[input, output] = tensor.shape().dims() else {
        return Err(RuntimeError::InvalidInput(format!(
            "GGML matrix {name} must be rank 2"
        )));
    };
    let source = tensor.as_f32()?;
    let mut weights = vec![0.0; source.len()];
    let head_dim = rope_head_dim(name, output, rope_heads)?;
    for (output_index, row) in source.chunks_exact(input).enumerate() {
        let column = head_dim.map_or(output_index, |dim| qk_channel(output_index, dim));
        for (input_index, &value) in row.iter().enumerate() {
            weights[input_index * output + column] = value;
        }
    }
    Ok(Tensor::from_f32(vec![input, output], weights)?)
}
