//! GPTQ packed-weight decoding.

use quantinfer_core::Tensor;
use quantinfer_format::SafeTensorInfo;

use super::config::GptqConfig;
use super::source::GptqTensorSource;

/// Decode a GPTQ matrix whose tensors use `prefix.qweight`, `prefix.qzeros`,
/// `prefix.scales`, and an optional `prefix.g_idx` name.
pub fn dequantize<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &GptqConfig,
) -> Result<Tensor, String> {
    dequantize_impl(source, prefix, config, None)
}

/// Decode a checkpoint with an explicit zero-point layout. Use this for
/// nonstandard transposed/direct tables whose shape is ambiguous. The ordinary
/// entry point gives the standard group-major packed layout precedence.
pub fn dequantize_with_zero_point_layout<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &GptqConfig,
    layout: ZeroPointLayout,
) -> Result<Tensor, String> {
    dequantize_impl(source, prefix, config, Some(layout))
}

fn dequantize_impl<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &GptqConfig,
    explicit_layout: Option<ZeroPointLayout>,
) -> Result<Tensor, String> {
    config.validate()?;
    let qweight_name = format!("{prefix}.qweight");
    let qzeros_name = format!("{prefix}.qzeros");
    let scales_name = format!("{prefix}.scales");
    let g_idx_name = format!("{prefix}.g_idx");

    let qweight_info = required_info(source, &qweight_name)?;
    if qweight_info.shape.len() != 2 {
        return Err(format!(
            "GPTQ tensor {qweight_name} must be rank 2, got shape {:?}",
            qweight_info.shape
        ));
    }
    let packed_input = qweight_info.shape[0];
    let output = qweight_info.shape[1];
    if packed_input == 0 || output == 0 {
        return Err(format!("GPTQ tensor {qweight_name} must be non-empty"));
    }
    let input = packed_input
        .checked_mul(config.pack_factor())
        .ok_or_else(|| format!("GPTQ tensor {qweight_name} input shape overflows usize"))?;
    let expected_qweight = packed_input
        .checked_mul(output)
        .ok_or_else(|| format!("GPTQ tensor {qweight_name} shape overflows usize"))?;
    let qweight = source.tensor_u32(&qweight_name)?;
    require_len(&qweight_name, qweight.len(), expected_qweight)?;

    let scales_info = required_info(source, &scales_name)?;
    if scales_info.shape.len() != 2 || scales_info.shape[0] == 0 || scales_info.shape[1] != output {
        return Err(format!(
            "GPTQ tensor {scales_name} must have shape [groups, {output}], got {:?}",
            scales_info.shape
        ));
    }
    let groups = scales_info.shape[0];
    let scales = source.tensor_f32(&scales_name)?;
    require_len(
        &scales_name,
        scales.len(),
        groups
            .checked_mul(output)
            .ok_or_else(|| format!("GPTQ tensor {scales_name} shape overflows usize"))?,
    )?;
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(format!(
            "GPTQ tensor {scales_name} contains a non-finite scale"
        ));
    }

    let zero_layout = source
        .tensor_info(&qzeros_name)
        .map(|info| {
            if let Some(layout) = explicit_layout {
                layout.validate_shape(&info.shape, groups, output, config.pack_factor())?;
                Ok(layout)
            } else {
                ZeroPointLayout::from_shape(&info.shape, groups, output, config.pack_factor())
            }
        })
        .transpose()?;
    if zero_layout.is_none() && !config.sym {
        return Err(format!(
            "GPTQ tensor {qzeros_name} is required for asymmetric quantization"
        ));
    }
    let qzeros = if let Some(layout) = zero_layout {
        let values = source.tensor_u32(&qzeros_name)?;
        require_len(
            &qzeros_name,
            values.len(),
            layout.element_count(groups, output, config.pack_factor()),
        )?;
        Some(values)
    } else {
        None
    };

    let g_idx = if let Some(info) = source.tensor_info(&g_idx_name) {
        if info.shape.as_slice() != [input] {
            return Err(format!(
                "GPTQ tensor {g_idx_name} must have shape [{input}], got {:?}",
                info.shape
            ));
        }
        let values = source.tensor_u32(&g_idx_name)?;
        require_len(&g_idx_name, values.len(), input)?;
        Some(values)
    } else {
        None
    };

    let mut decoded = Vec::with_capacity(
        input
            .checked_mul(output)
            .ok_or_else(|| "GPTQ decoded matrix shape overflows usize".to_owned())?,
    );
    for input_index in 0..input {
        let group = g_idx
            .as_ref()
            .map_or(input_index / config.group_size, |indices| {
                usize::try_from(indices[input_index]).unwrap_or(usize::MAX)
            });
        if group >= groups {
            return Err(format!(
                "GPTQ group index {group} for input row {input_index} is outside {groups} groups"
            ));
        }
        let input_word = input_index / config.pack_factor();
        let input_shift = (input_index % config.pack_factor()) * config.bits;
        for output_index in 0..output {
            let packed = qweight[input_word * output + output_index];
            let code = (packed >> input_shift) & config.code_mask();
            let zero = if let (Some(layout), Some(values)) = (zero_layout, qzeros.as_ref()) {
                layout.value(values, group, output_index, groups, output, config)
            } else {
                1_u32 << (config.bits - 1)
            };
            let scale = scales[group * output + output_index];
            let value = scale * (code as f32 - zero as f32);
            if !value.is_finite() {
                return Err(format!(
                    "GPTQ {prefix} reconstructed weight [{input_index}, {output_index}] exceeds finite F32 range"
                ));
            }
            decoded.push(value);
        }
    }
    Tensor::from_f32(vec![input, output], decoded).map_err(|error| error.to_string())
}

fn required_info<S: GptqTensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Result<SafeTensorInfo, String> {
    source
        .tensor_info(name)
        .ok_or_else(|| format!("missing GPTQ tensor {name}"))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "GPTQ tensor {name} contains {actual} values, expected {expected}"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPointLayout {
    /// Standard GPTQ: `[groups, ceil(output / pack_factor)]`.
    GroupPacked,
    /// Transposed packed table: `[ceil(output / pack_factor), groups]`.
    PackedGroup,
    /// One stored code per value: `[groups, output]`.
    GroupDirect,
    /// Transposed direct table: `[output, groups]`.
    DirectGroup,
}

impl ZeroPointLayout {
    fn from_shape(
        shape: &[usize],
        groups: usize,
        output: usize,
        pack_factor: usize,
    ) -> Result<Self, String> {
        if shape.len() != 2 {
            return Err(format!("GPTQ qzeros must be rank 2, got shape {shape:?}"));
        }
        let packed_output = output.div_ceil(pack_factor);
        if shape == [groups, packed_output] {
            Ok(Self::GroupPacked)
        } else if shape == [packed_output, groups] {
            Ok(Self::PackedGroup)
        } else if shape == [groups, output] {
            Ok(Self::GroupDirect)
        } else if shape == [output, groups] {
            Ok(Self::DirectGroup)
        } else {
            Err(format!(
                "GPTQ qzeros shape {shape:?} does not match groups={groups}, output={output}, packed_output={packed_output}"
            ))
        }
    }

    fn validate_shape(
        self,
        shape: &[usize],
        groups: usize,
        output: usize,
        pack_factor: usize,
    ) -> Result<(), String> {
        let packed = output.div_ceil(pack_factor);
        let expected = match self {
            Self::GroupPacked => [groups, packed],
            Self::PackedGroup => [packed, groups],
            Self::GroupDirect => [groups, output],
            Self::DirectGroup => [output, groups],
        };
        if shape != expected {
            return Err(format!(
                "GPTQ qzeros shape {shape:?} does not match explicit {self:?} layout {expected:?}"
            ));
        }
        Ok(())
    }

    fn element_count(self, groups: usize, output: usize, pack_factor: usize) -> usize {
        match self {
            Self::GroupPacked | Self::PackedGroup => groups * output.div_ceil(pack_factor),
            Self::GroupDirect | Self::DirectGroup => groups * output,
        }
    }

    fn value(
        self,
        values: &[u32],
        group: usize,
        output: usize,
        groups: usize,
        output_count: usize,
        config: &GptqConfig,
    ) -> u32 {
        let pack_factor = config.pack_factor();
        let packed_output = output_count.div_ceil(pack_factor);
        let (word, shift) = match self {
            Self::GroupPacked => (
                values[group * packed_output + output / pack_factor],
                (output % pack_factor) * config.bits,
            ),
            Self::PackedGroup => (
                values[(output / pack_factor) * groups + group],
                (output % pack_factor) * config.bits,
            ),
            Self::GroupDirect => (values[group * output_count + output], 0),
            Self::DirectGroup => (values[output * groups + group], 0),
        };
        // GPTQ-v1 stores (zero - 1) modulo the code width; v2 uses offset 0.
        // Restore in the same finite-width domain, including stored max code.
        ((word >> shift) & config.code_mask()).wrapping_add_signed(config.zero_offset)
            & config.code_mask()
    }
}
#[cfg(test)]
mod tests;
