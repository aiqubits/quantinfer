//! Preserve codes/scales while projecting native rows into a source profile.

use quantinfer::runtime::LinearWeight;
use quantinfer_core::{QuantizedTensor, QuantizedTensor4};
use quantinfer_format::{qk_channel, GgufReader, GgufWriter};
use quantinfer_quant::QuantizedData;
use std::borrow::Cow;

pub(super) fn encoded_projection(
    data: &QuantizedData,
    name: &str,
    rope_head_dim: Option<usize>,
) -> Result<Vec<u8>, String> {
    let &[input, output] = data.shape().dims() else {
        return Err("projection must be a matrix".into());
    };
    if input == 0 || !input.is_multiple_of(32) || output == 0 {
        return Err("packed projection requires a positive input width divisible by 32".into());
    }
    let mut writer = GgufWriter::new();
    match data {
        QuantizedData::Q4_0(tensor) => {
            let tensor = if let Some(dimension) = rope_head_dim {
                Cow::Owned(
                    QuantizedTensor4::from_blocks(
                        tensor.shape().clone(),
                        export_qk_rows(tensor.blocks(), input / 32, dimension)?,
                    )
                    .map_err(|e| e.to_string())?,
                )
            } else {
                Cow::Borrowed(tensor)
            };
            writer.add_q4_0_tensor(name, &tensor);
        }
        QuantizedData::Q8_0(tensor) => {
            let tensor = if let Some(dimension) = rope_head_dim {
                Cow::Owned(
                    QuantizedTensor::from_blocks(
                        tensor.shape().clone(),
                        export_qk_rows(tensor.blocks(), input / 32, dimension)?,
                    )
                    .map_err(|e| e.to_string())?,
                )
            } else {
                Cow::Borrowed(tensor)
            };
            writer.add_q8_0_tensor(name, &tensor);
        }
        _ => return Err("Calibrated publication expects Q4_0 or Q8_0 output-row blocks".into()),
    }
    let reader =
        GgufReader::open(writer.finish().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    reader
        .tensor_bytes(name)
        .map(|bytes| bytes.to_vec())
        .map_err(|e| e.to_string())
}

/// Inverse of the runtime GGML Llama Q/K row import. This moves complete rows;
/// neither the code points nor their F16 scale grids are changed.
fn export_qk_rows<T: Clone>(
    blocks: &[T],
    row_blocks: usize,
    head_dim: usize,
) -> Result<Vec<T>, String> {
    if row_blocks == 0
        || !blocks.len().is_multiple_of(row_blocks)
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !(blocks.len() / row_blocks).is_multiple_of(head_dim)
    {
        return Err("invalid Llama Q/K channel permutation".into());
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(blocks.len())
        .map_err(|e| format!("cannot allocate Q/K row permutation: {e}"))?;
    for row in 0..blocks.len() / row_blocks {
        let source = qk_channel(row, head_dim) * row_blocks;
        result.extend_from_slice(&blocks[source..source + row_blocks]);
    }
    Ok(result)
}

pub(super) fn runtime_weight(data: QuantizedData) -> Result<LinearWeight, String> {
    match data {
        QuantizedData::Q4_0(tensor) => Ok(LinearWeight::Q4_0OutputRows(tensor)),
        QuantizedData::Q8_0(tensor) => Ok(LinearWeight::Q8_0OutputRows(tensor)),
        _ => Err("Calibrated model replacement expects signed output-row storage".into()),
    }
}
