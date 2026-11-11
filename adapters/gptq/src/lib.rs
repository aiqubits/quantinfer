//! GPTQ SafeTensors importer.
//!
//! The importer decodes the ordinary affine GPTQ representation into a
//! row-major F32 tensor. It keeps the packed representation at the format
//! boundary so runtimes can choose their own execution layout.

#![forbid(unsafe_code)]

use quantinfer_core::Tensor;
use quantinfer_format::{SafeTensorFile, SafeTensorInfo, SafeTensorShards};
use serde::Deserialize;

/// Configuration needed to decode one GPTQ linear weight.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GptqConfig {
    /// Number of bits in each packed weight code.
    pub bits: usize,
    /// Number of input rows sharing one scale/zero-point group.
    pub group_size: usize,
    /// Symmetric sources may omit `qzeros`; the midpoint code is used then.
    #[serde(default)]
    pub sym: bool,
    /// Stored GPTQ zero points are normally one less than the logical code.
    /// This remains explicit for checkpoints using a different convention.
    #[serde(default = "default_zero_offset")]
    pub zero_offset: i32,
}

impl GptqConfig {
    pub fn new(bits: usize, group_size: usize, sym: bool) -> Self {
        Self {
            bits,
            group_size,
            sym,
            zero_offset: default_zero_offset(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(format!("GPTQ supports 2, 4, or 8 bits, got {}", self.bits));
        }
        if self.group_size == 0 {
            return Err("GPTQ group_size must be greater than zero".to_owned());
        }
        if self.bits > 32 || !32usize.is_multiple_of(self.bits) {
            return Err(format!(
                "GPTQ bits must divide a 32-bit packed word, got {}",
                self.bits
            ));
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

const fn default_zero_offset() -> i32 {
    1
}

/// A source of GPTQ tensors. Both one-file and indexed SafeTensors models are
/// supported without merging their payloads in memory.
pub trait GptqTensorSource {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo>;
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String>;
    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String>;
}

impl GptqTensorSource for SafeTensorFile {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo> {
        self.tensors().get(name).cloned()
    }

    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        SafeTensorFile::tensor_f32(self, name).map_err(|error| error.to_string())
    }

    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String> {
        SafeTensorFile::tensor_u32(self, name).map_err(|error| error.to_string())
    }
}

impl GptqTensorSource for SafeTensorShards {
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
}

/// Decode a GPTQ matrix whose tensors use `prefix.qweight`, `prefix.qzeros`,
/// `prefix.scales`, and an optional `prefix.g_idx` name.
pub fn dequantize<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &GptqConfig,
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
        .map(|info| ZeroLayout::from_shape(&info.shape, groups, output, config.pack_factor()))
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
            decoded.push(scale * (code as f32 - zero as f32));
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
enum ZeroLayout {
    GroupPacked,
    PackedGroup,
    GroupDirect,
    DirectGroup,
}

impl ZeroLayout {
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
        if shape == [packed_output, groups] {
            Ok(Self::PackedGroup)
        } else if shape == [groups, packed_output] {
            Ok(Self::GroupPacked)
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
        ((word >> shift) & config.code_mask()).saturating_add_signed(config.zero_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{dequantize, GptqConfig};
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

    fn floats(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn decodes_standard_packed_gptq_matrix() {
        let bits = 4;
        let mut qweight = [0_u32; 2];
        for (input, code) in [2_u32, 3, 4, 5, 6, 7, 8, 9].into_iter().enumerate() {
            qweight[0] |= code << (input * bits);
        }
        for (input, code) in [1_u32, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
            qweight[1] |= code << (input * bits);
        }
        let file = fixture(&[
            ("layer.qweight", "I32", vec![1, 2], words(&qweight)),
            ("layer.qzeros", "I32", vec![1, 2], words(&[33, 33])),
            (
                "layer.scales",
                "F32",
                vec![2, 2],
                floats(&[1.0, 2.0, 3.0, 4.0]),
            ),
        ]);
        let tensor = dequantize(&file, "layer", &GptqConfig::new(bits, 4, false)).unwrap();
        assert_eq!(tensor.shape().dims(), &[8, 2]);
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[
                0.0, -4.0, 1.0, -2.0, 2.0, 0.0, 3.0, 2.0, 12.0, 8.0, 15.0, 12.0, 18.0, 16.0, 21.0,
                20.0,
            ]
        );
    }

    #[test]
    fn uses_g_idx_and_symmetric_midpoint_when_qzeros_are_absent() {
        let mut qword = 0_u32;
        for input in 0..8 {
            qword |= (input as u32) << (input * 4);
        }
        let g_idx: Vec<u32> = vec![1; 8];
        let file = fixture(&[
            ("layer.qweight", "U32", vec![1, 1], words(&[qword])),
            ("layer.scales", "F32", vec![2, 1], floats(&[1.0, 2.0])),
            ("layer.g_idx", "I32", vec![8], words(&g_idx)),
        ]);
        let tensor = dequantize(&file, "layer", &GptqConfig::new(4, 4, true)).unwrap();
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[-16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0]
        );
    }

    #[test]
    fn rejects_invalid_bits_and_shapes() {
        assert!(GptqConfig::new(3, 4, false).validate().is_err());
    }
}
