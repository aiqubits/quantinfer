//! Decode GGML tensor payloads into runtime tensors.

use quantinfer_core::Tensor;
use quantinfer_format::{GgmlType, GgufReader};
use quantinfer_quant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};

use crate::{Result, RuntimeError};

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
