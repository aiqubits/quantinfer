//! LLM Compressor `compressed-tensors` WNA16 importer.
//!
//! The adapter handles the integer packed weight-only layout used by
//! compressed-tensors: `weight_packed` is packed along the input axis,
//! `weight_scale` is indexed by output and input group, and the optional
//! `weight_zero_point` is packed along the output axis.

#![forbid(unsafe_code)]

use quantinfer_core::Tensor;
use quantinfer_format::{SafeTensorFile, SafeTensorInfo, SafeTensorShards};
use serde::Deserialize;

/// Configuration for one compressed-tensors WNA16 matrix.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LlmCompressorConfig {
    pub bits: usize,
    pub group_size: usize,
    #[serde(default)]
    pub symmetric: bool,
    /// Some exporters store a zero point with a fixed integer offset.
    #[serde(default)]
    pub zero_offset: i32,
}

impl LlmCompressorConfig {
    pub fn new(bits: usize, group_size: usize, symmetric: bool) -> Self {
        Self {
            bits,
            group_size,
            symmetric,
            zero_offset: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(format!(
                "compressed-tensors supports 2, 4, or 8 bits, got {}",
                self.bits
            ));
        }
        if self.group_size == 0 {
            return Err("compressed-tensors group_size must be greater than zero".to_owned());
        }
        Ok(())
    }

    fn pack_factor(&self) -> usize {
        32 / self.bits
    }

    fn code_mask(&self) -> u32 {
        (1_u32 << self.bits) - 1
    }
}

pub trait LlmCompressorTensorSource {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo>;
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String>;
    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String>;
    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String>;
}

impl LlmCompressorTensorSource for SafeTensorFile {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo> {
        self.tensors().get(name).cloned()
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        SafeTensorFile::tensor_f32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        SafeTensorFile::tensor_u32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        SafeTensorFile::tensor_u64(self, name).map_err(|error| error.to_string())
    }
}

impl LlmCompressorTensorSource for SafeTensorShards {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo> {
        self.tensors()
            .find_map(|(tensor_name, info)| (tensor_name == name).then(|| info.clone()))
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        SafeTensorShards::tensor_f32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        SafeTensorShards::tensor_u32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        SafeTensorShards::tensor_u64(self, name).map_err(|error| error.to_string())
    }
}

/// Decode `prefix.weight_packed`, `prefix.weight_scale`, and an optional
/// `prefix.weight_zero_point` into a row-major `[input, output]` F32 tensor.
pub fn dequantize<S: LlmCompressorTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &LlmCompressorConfig,
) -> Result<Tensor, String> {
    config.validate()?;
    let weight_name = format!("{prefix}.weight_packed");
    let scale_name = format!("{prefix}.weight_scale");
    let zero_name = format!("{prefix}.weight_zero_point");
    let shape_name = format!("{prefix}.weight_shape");
    let g_idx_name = format!("{prefix}.weight_g_idx");

    let weight_info = required_info(source, &weight_name)?;
    if weight_info.shape.len() != 2 {
        return Err(format!(
            "compressed-tensors {weight_name} must be rank 2, got shape {:?}",
            weight_info.shape
        ));
    }
    let output = weight_info.shape[0];
    let packed_input = weight_info.shape[1];
    if output == 0 || packed_input == 0 {
        return Err(format!(
            "compressed-tensors {weight_name} must be non-empty"
        ));
    }
    let input = packed_input
        .checked_mul(config.pack_factor())
        .ok_or_else(|| format!("compressed-tensors {weight_name} shape overflows usize"))?;
    let weights = source.tensor_u32(&weight_name)?;
    require_len(
        &weight_name,
        weights.len(),
        output
            .checked_mul(packed_input)
            .ok_or_else(|| format!("compressed-tensors {weight_name} shape overflows usize"))?,
    )?;

    let scale_info = required_info(source, &scale_name)?;
    if scale_info.shape.len() != 2 || scale_info.shape[0] != output || scale_info.shape[1] == 0 {
        return Err(format!(
            "compressed-tensors {scale_name} must have shape [{output}, groups], got {:?}",
            scale_info.shape
        ));
    }
    let groups = scale_info.shape[1];
    let scales = source.tensor_f32(&scale_name)?;
    require_len(
        &scale_name,
        scales.len(),
        output
            .checked_mul(groups)
            .ok_or_else(|| format!("compressed-tensors {scale_name} shape overflows usize"))?,
    )?;
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(format!(
            "compressed-tensors {scale_name} contains a non-finite scale"
        ));
    }

    if let Some(info) = source.tensor_info(&shape_name) {
        if info.shape.as_slice() != [2] {
            return Err(format!(
                "compressed-tensors {shape_name} must have shape [2], got {:?}",
                info.shape
            ));
        }
        let shape = source.tensor_u64(&shape_name)?;
        require_len(&shape_name, shape.len(), 2)?;
        let expected = [input as u64, output as u64];
        let transposed = [output as u64, input as u64];
        if shape != expected && shape != transposed {
            return Err(format!(
                "compressed-tensors {shape_name} records {:?}, expected {:?} or {:?}",
                shape, expected, transposed
            ));
        }
    }

    let zero_layout = source
        .tensor_info(&zero_name)
        .map(|info| ZeroLayout::from_shape(&info.shape, output, groups, config.pack_factor()))
        .transpose()?;
    if zero_layout.is_none() && !config.symmetric {
        return Err(format!(
            "compressed-tensors {zero_name} is required for asymmetric quantization"
        ));
    }
    let zeros = if let Some(layout) = zero_layout {
        let values = source.tensor_u32(&zero_name)?;
        require_len(
            &zero_name,
            values.len(),
            layout.element_count(output, groups, config.pack_factor()),
        )?;
        Some(values)
    } else {
        None
    };

    let g_idx = if let Some(info) = source.tensor_info(&g_idx_name) {
        if info.shape.as_slice() != [input] {
            return Err(format!(
                "compressed-tensors {g_idx_name} must have shape [{input}], got {:?}",
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
            .ok_or_else(|| "compressed-tensors decoded shape overflows usize".to_owned())?,
    );
    for input_index in 0..input {
        let group = g_idx
            .as_ref()
            .map_or(input_index / config.group_size, |indices| {
                usize::try_from(indices[input_index]).unwrap_or(usize::MAX)
            });
        if group >= groups {
            return Err(format!(
                "compressed-tensors group index {group} for input row {input_index} is outside {groups} groups"
            ));
        }
        let input_word = input_index / config.pack_factor();
        let shift = (input_index % config.pack_factor()) * config.bits;
        for output_index in 0..output {
            let packed = weights[output_index * packed_input + input_word];
            let code = (packed >> shift) & config.code_mask();
            let zero = if let (Some(layout), Some(values)) = (zero_layout, zeros.as_ref()) {
                layout.value(values, output_index, group, groups, config)
            } else {
                1_u32 << (config.bits - 1)
            };
            decoded.push(scales[output_index * groups + group] * (code as f32 - zero as f32));
        }
    }
    Tensor::from_f32(vec![input, output], decoded).map_err(|error| error.to_string())
}

fn required_info<S: LlmCompressorTensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Result<SafeTensorInfo, String> {
    source
        .tensor_info(name)
        .ok_or_else(|| format!("missing compressed-tensors tensor {name}"))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "compressed-tensors tensor {name} contains {actual} values, expected {expected}"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZeroLayout {
    PackedOutputGroups,
    DirectOutputGroups,
}

impl ZeroLayout {
    fn from_shape(
        shape: &[usize],
        output: usize,
        groups: usize,
        pack_factor: usize,
    ) -> Result<Self, String> {
        if shape == [output.div_ceil(pack_factor), groups] {
            Ok(Self::PackedOutputGroups)
        } else if shape == [output, groups] {
            Ok(Self::DirectOutputGroups)
        } else {
            Err(format!(
                "compressed-tensors zero-point shape {shape:?} does not match output={output}, groups={groups}"
            ))
        }
    }

    fn element_count(self, output: usize, groups: usize, pack_factor: usize) -> usize {
        match self {
            Self::PackedOutputGroups => output.div_ceil(pack_factor) * groups,
            Self::DirectOutputGroups => output * groups,
        }
    }

    fn value(
        self,
        values: &[u32],
        output_index: usize,
        group: usize,
        groups: usize,
        config: &LlmCompressorConfig,
    ) -> u32 {
        let (word, shift) = match self {
            Self::PackedOutputGroups => (
                values[(output_index / config.pack_factor()) * groups + group],
                (output_index % config.pack_factor()) * config.bits,
            ),
            Self::DirectOutputGroups => (values[output_index * groups + group], 0),
        };
        ((word >> shift) & config.code_mask()).saturating_add_signed(config.zero_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{dequantize, LlmCompressorConfig};
    use quantinfer_format::SafeTensorFile;
    use serde_json::json;

    fn fixture(entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> SafeTensorFile {
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, payload) in entries {
            let end = offset + payload.len();
            header.insert(
                (*name).to_owned(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&json!(header)).unwrap();
        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for (_, _, _, payload) in entries {
            bytes.extend_from_slice(payload);
        }
        SafeTensorFile::open(bytes).unwrap()
    }

    fn words(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u64s(values: &[u64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn decodes_compressed_tensors_w4a16_layout() {
        let mut first = 0_u32;
        let mut second = 0_u32;
        for input in 0..8 {
            first |= (input as u32) << (input * 4);
            second |= ((input as u32 + 1) & 0xF) << (input * 4);
        }
        let file = fixture(&[
            (
                "layer.weight_packed",
                "I32",
                vec![2, 1],
                words(&[first, second]),
            ),
            (
                "layer.weight_scale",
                "F32",
                vec![2, 2],
                floats(&[1.0, 2.0, 3.0, 4.0]),
            ),
            (
                "layer.weight_zero_point",
                "I32",
                vec![1, 2],
                words(&[0x21, 0x21]),
            ),
            ("layer.weight_shape", "I64", vec![2], u64s(&[8, 2])),
        ]);
        let tensor = dequantize(&file, "layer", &LlmCompressorConfig::new(4, 4, false)).unwrap();
        assert_eq!(tensor.shape().dims(), &[8, 2]);
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[
                -1.0, -3.0, 0.0, 0.0, 1.0, 3.0, 2.0, 6.0, 6.0, 12.0, 8.0, 16.0, 10.0, 20.0, 12.0,
                24.0
            ]
        );
    }

    #[test]
    fn supports_symmetric_checkpoint_without_zero_points() {
        let file = fixture(&[
            (
                "layer.weight_packed",
                "U32",
                vec![1, 1],
                words(&[0x7654_3210]),
            ),
            ("layer.weight_scale", "F32", vec![1, 1], floats(&[2.0])),
        ]);
        let tensor = dequantize(&file, "layer", &LlmCompressorConfig::new(4, 8, true)).unwrap();
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[-16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0]
        );
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(LlmCompressorConfig::new(3, 4, false).validate().is_err());
        assert!(LlmCompressorConfig::new(4, 0, false).validate().is_err());
    }
}
