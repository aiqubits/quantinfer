//! AutoAWQ SafeTensors importer.
//!
//! AutoAWQ stores 4-bit values packed along the output dimension and uses an
//! interleaved bit order inside each 32-bit word. This adapter decodes that
//! representation into a row-major F32 tensor at the format boundary.

#![forbid(unsafe_code)]

use quantinfer_core::Tensor;
use quantinfer_format::{SafeTensorFile, SafeTensorInfo, SafeTensorShards};
use serde::Deserialize;

const AWQ_BITS: usize = 4;
const AWQ_PACK_FACTOR: usize = 8;
// Raw AWQ slots for logical output indices 0..7.
const AWQ_RAW_SLOT: [usize; AWQ_PACK_FACTOR] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Configuration needed to decode one AutoAWQ linear weight.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct AwqConfig {
    /// AutoAWQ currently uses four-bit words.
    pub bits: usize,
    /// Number of input rows sharing one scale/zero-point group.
    pub group_size: usize,
    /// Symmetric sources may omit `qzeros`; the midpoint code is used then.
    #[serde(default)]
    pub sym: bool,
    /// Offset applied to packed zero-point codes. AutoAWQ uses zero by default.
    #[serde(default)]
    pub zero_offset: i32,
}

impl AwqConfig {
    pub fn new(group_size: usize, sym: bool) -> Self {
        Self {
            bits: AWQ_BITS,
            group_size,
            sym,
            zero_offset: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.bits != AWQ_BITS {
            return Err(format!(
                "AutoAWQ SafeTensors currently supports {} bits, got {}",
                AWQ_BITS, self.bits
            ));
        }
        if self.group_size == 0 {
            return Err("AWQ group_size must be greater than zero".to_owned());
        }
        Ok(())
    }
}

/// A source of AWQ tensors. Both one-file and indexed SafeTensors models are
/// supported without merging their payloads in memory.
pub trait AwqTensorSource {
    fn tensor_info(&self, name: &str) -> Option<SafeTensorInfo>;
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String>;
    fn tensor_u32(&self, name: &str) -> Result<Vec<u32>, String>;
}

impl AwqTensorSource for SafeTensorFile {
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

impl AwqTensorSource for SafeTensorShards {
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

/// Decode a matrix whose tensors use `prefix.qweight`, `prefix.qzeros`,
/// `prefix.scales`, and an optional `prefix.g_idx` name.
pub fn dequantize<S: AwqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &AwqConfig,
) -> Result<Tensor, String> {
    config.validate()?;
    let qweight_name = format!("{prefix}.qweight");
    let qzeros_name = format!("{prefix}.qzeros");
    let scales_name = format!("{prefix}.scales");
    let g_idx_name = format!("{prefix}.g_idx");

    let qweight_info = required_info(source, &qweight_name)?;
    if qweight_info.shape.len() != 2 {
        return Err(format!(
            "AWQ tensor {qweight_name} must be rank 2, got shape {:?}",
            qweight_info.shape
        ));
    }
    let input = qweight_info.shape[0];
    let packed_output = qweight_info.shape[1];
    if input == 0 || packed_output == 0 {
        return Err(format!("AWQ tensor {qweight_name} must be non-empty"));
    }
    let qweight = source.tensor_u32(&qweight_name)?;
    require_len(
        &qweight_name,
        qweight.len(),
        input
            .checked_mul(packed_output)
            .ok_or_else(|| format!("AWQ tensor {qweight_name} shape overflows usize"))?,
    )?;

    let scales_info = required_info(source, &scales_name)?;
    if scales_info.shape.len() != 2 || scales_info.shape[0] == 0 || scales_info.shape[1] == 0 {
        return Err(format!(
            "AWQ tensor {scales_name} must have shape [groups, output], got {:?}",
            scales_info.shape
        ));
    }
    let groups = scales_info.shape[0];
    let output = scales_info.shape[1];
    let expected_packed_output = output.div_ceil(AWQ_PACK_FACTOR);
    if packed_output != expected_packed_output {
        return Err(format!(
            "AWQ tensor {qweight_name} has {packed_output} packed output columns, expected {expected_packed_output} for output={output}"
        ));
    }
    let scales = source.tensor_f32(&scales_name)?;
    require_len(
        &scales_name,
        scales.len(),
        groups
            .checked_mul(output)
            .ok_or_else(|| format!("AWQ tensor {scales_name} shape overflows usize"))?,
    )?;
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err(format!(
            "AWQ tensor {scales_name} contains a non-finite scale"
        ));
    }

    let zero_layout = source
        .tensor_info(&qzeros_name)
        .map(|info| ZeroLayout::from_shape(&info.shape, groups, output, packed_output))
        .transpose()?;
    if zero_layout.is_none() && !config.sym {
        return Err(format!(
            "AWQ tensor {qzeros_name} is required for asymmetric quantization"
        ));
    }
    let qzeros = if let Some(layout) = zero_layout {
        let values = source.tensor_u32(&qzeros_name)?;
        require_len(
            &qzeros_name,
            values.len(),
            layout.element_count(groups, output, packed_output),
        )?;
        Some(values)
    } else {
        None
    };

    let g_idx = if let Some(info) = source.tensor_info(&g_idx_name) {
        if info.shape.as_slice() != [input] {
            return Err(format!(
                "AWQ tensor {g_idx_name} must have shape [{input}], got {:?}",
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
            .ok_or_else(|| "AWQ decoded matrix shape overflows usize".to_owned())?,
    );
    for input_index in 0..input {
        let group = g_idx
            .as_ref()
            .map_or(input_index / config.group_size, |indices| {
                usize::try_from(indices[input_index]).unwrap_or(usize::MAX)
            });
        if group >= groups {
            return Err(format!(
                "AWQ group index {group} for input row {input_index} is outside {groups} groups"
            ));
        }
        for output_index in 0..output {
            let word = qweight[input_index * packed_output + output_index / AWQ_PACK_FACTOR];
            let slot = AWQ_RAW_SLOT[output_index % AWQ_PACK_FACTOR];
            let code = (word >> (slot * AWQ_BITS)) & 0xF;
            let zero = if let (Some(layout), Some(values)) = (zero_layout, qzeros.as_ref()) {
                layout.value(values, group, output_index, output, packed_output, config)
            } else {
                1_u32 << (AWQ_BITS - 1)
            };
            decoded.push(scales[group * output + output_index] * (code as f32 - zero as f32));
        }
    }
    Tensor::from_f32(vec![input, output], decoded).map_err(|error| error.to_string())
}

fn required_info<S: AwqTensorSource + ?Sized>(
    source: &S,
    name: &str,
) -> Result<SafeTensorInfo, String> {
    source
        .tensor_info(name)
        .ok_or_else(|| format!("missing AWQ tensor {name}"))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "AWQ tensor {name} contains {actual} values, expected {expected}"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZeroLayout {
    GroupPacked,
    GroupDirect,
}

impl ZeroLayout {
    fn from_shape(
        shape: &[usize],
        groups: usize,
        output: usize,
        packed_output: usize,
    ) -> Result<Self, String> {
        if shape == [groups, packed_output] {
            Ok(Self::GroupPacked)
        } else if shape == [groups, output] {
            Ok(Self::GroupDirect)
        } else {
            Err(format!(
                "AWQ qzeros shape {shape:?} does not match groups={groups}, output={output}, packed_output={packed_output}"
            ))
        }
    }

    fn element_count(self, groups: usize, output: usize, packed_output: usize) -> usize {
        match self {
            Self::GroupPacked => groups * packed_output,
            Self::GroupDirect => groups * output,
        }
    }

    fn value(
        self,
        values: &[u32],
        group: usize,
        output_index: usize,
        output: usize,
        packed_output: usize,
        config: &AwqConfig,
    ) -> u32 {
        let (word, shift) = match self {
            Self::GroupPacked => (
                values[group * packed_output + output_index / AWQ_PACK_FACTOR],
                AWQ_RAW_SLOT[output_index % AWQ_PACK_FACTOR] * AWQ_BITS,
            ),
            Self::GroupDirect => (values[group * output + output_index], 0),
        };
        ((word >> shift) & 0xF).saturating_add_signed(config.zero_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{dequantize, AwqConfig, AWQ_RAW_SLOT};
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
    fn decodes_awq_interleaved_output_packing() {
        let mut first = 0_u32;
        let mut second = 0_u32;
        for (logical, slot) in AWQ_RAW_SLOT.iter().copied().enumerate() {
            first |= (logical as u32) << (slot * 4);
            second |= ((logical as u32 + 8) & 0xF) << (slot * 4);
        }
        let zero = (0..8).fold(0_u32, |word, logical| {
            word | 2_u32 << (AWQ_RAW_SLOT[logical] * 4)
        });
        let file = fixture(&[
            ("layer.qweight", "I32", vec![2, 1], words(&[first, second])),
            ("layer.qzeros", "I32", vec![1, 1], words(&[zero])),
            (
                "layer.scales",
                "F32",
                vec![1, 8],
                floats(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ),
        ]);
        let tensor = dequantize(&file, "layer", &AwqConfig::new(2, false)).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 8]);
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[
                -2.0, -2.0, 0.0, 4.0, 10.0, 18.0, 28.0, 40.0, 6.0, 14.0, 24.0, 36.0, 50.0, 66.0,
                84.0, 104.0,
            ]
        );
    }

    #[test]
    fn supports_symmetric_awq_without_qzeros() {
        let qword = AWQ_RAW_SLOT
            .iter()
            .enumerate()
            .fold(0_u32, |word, (logical, slot)| {
                word | (logical as u32) << (slot * 4)
            });
        let file = fixture(&[
            ("layer.qweight", "U32", vec![1, 1], words(&[qword])),
            ("layer.scales", "F32", vec![1, 8], floats(&[1.0; 8])),
        ]);
        let tensor = dequantize(&file, "layer", &AwqConfig::new(1, true)).unwrap();
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]
        );
    }

    #[test]
    fn rejects_non_four_bit_config() {
        let mut config = AwqConfig::new(32, false);
        config.bits = 8;
        assert!(config.validate().is_err());
    }
}
