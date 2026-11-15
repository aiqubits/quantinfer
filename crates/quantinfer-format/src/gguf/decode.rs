//! GGUF tensor payload decoding into core tensor representations.

use half::{bf16, f16};
use quantinfer_core::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};

use super::reader::PackedTensorReader;
use super::types::{FormatError, GgmlType};
use super::Result;

impl PackedTensorReader {
    pub fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let info = self.tensor_info(name)?;
        let bytes = self.tensor_bytes(name)?;
        let values: Vec<f32> = match info.dtype {
            GgmlType::F32 => {
                let (chunks, remainder) = bytes.as_chunks::<4>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF F32 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect()
            }
            GgmlType::F16 => {
                let (chunks, remainder) = bytes.as_chunks::<2>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF F16 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
                    .collect()
            }
            GgmlType::BF16 => {
                let (chunks, remainder) = bytes.as_chunks::<2>();
                if !remainder.is_empty() {
                    return Err(FormatError::Invalid(format!(
                        "GGUF BF16 tensor {name} has an invalid byte length"
                    )));
                }
                chunks
                    .iter()
                    .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                    .collect()
            }
            GgmlType::Q8_0 => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q8_0, expected floating point"
                )))
            }
            GgmlType::Q4_0 => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q4_0, expected floating point"
                )))
            }
            GgmlType::Q4_K => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has type Q4_K, expected floating point"
                )))
            }
            GgmlType::Q5_K | GgmlType::Q6_K => {
                return Err(FormatError::Invalid(format!(
                    "GGUF tensor {name} has a K-quant type, expected floating point"
                )))
            }
        };
        if values.len() != info.shape.iter().product::<usize>() {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has an invalid byte length"
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} contains non-finite values"
            )));
        }
        Ok(Tensor::from_f32(info.shape.clone(), values)?)
    }

    pub fn tensor_q8_0(&self, name: &str) -> Result<QuantizedTensor> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q8_0 {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q8_0",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let block_size = std::mem::size_of::<f16>() + Q8_0_BLOCK_SIZE;
        if !bytes.len().is_multiple_of(block_size) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q8_0 tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / block_size);
        for chunk in bytes.chunks_exact(block_size) {
            let scale = f16::from_le_bytes(
                chunk[..std::mem::size_of::<f16>()]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q8_0 scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q8_0 tensor {name} contains a non-finite scale"
                )));
            }
            let mut values = [0_i8; Q8_0_BLOCK_SIZE];
            for (index, value) in chunk[std::mem::size_of::<f16>()..]
                .iter()
                .copied()
                .enumerate()
            {
                values[index] = i8::from_le_bytes([value]);
            }
            blocks.push(Q8_0Block { scale, values });
        }
        Ok(QuantizedTensor::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q4_0(&self, name: &str) -> Result<QuantizedTensor4> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q4_0 {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q4_0",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let legacy = self.uses_legacy_quant_block_layout()?;
        let block_size = std::mem::size_of::<f16>() + Q4_0_BLOCK_SIZE / 2;
        if !bytes.len().is_multiple_of(block_size) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q4_0 tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / block_size);
        for chunk in bytes.chunks_exact(block_size) {
            let scale = f16::from_le_bytes(
                chunk[..std::mem::size_of::<f16>()]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_0 scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q4_0 tensor {name} contains a non-finite scale"
                )));
            }
            let mut values = [0_u8; Q4_0_BLOCK_SIZE];
            for (index, packed) in chunk[std::mem::size_of::<f16>()..]
                .iter()
                .copied()
                .enumerate()
            {
                let (low, high) = if legacy {
                    (index * 2, index * 2 + 1)
                } else {
                    (index, index + Q4_0_BLOCK_SIZE / 2)
                };
                values[low] = packed & 0x0f;
                values[high] = packed >> 4;
            }
            blocks.push(Q4_0Block { scale, values });
        }
        Ok(QuantizedTensor4::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q4_k(&self, name: &str) -> Result<QuantizedTensorQ4K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q4_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q4_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if !bytes.len().is_multiple_of(144) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q4_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 144);
        for chunk in bytes.as_chunks::<144>().0 {
            let scale = f16::from_le_bytes(
                chunk[..2]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_K scale".to_owned()))?,
            )
            .to_f32();
            let min_scale = f16::from_le_bytes(
                chunk[2..4]
                    .try_into()
                    .map_err(|_| FormatError::Invalid("invalid Q4_K minimum scale".to_owned()))?,
            )
            .to_f32();
            if !scale.is_finite() || !min_scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q4_K tensor {name} contains a non-finite scale"
                )));
            }
            let mut scales = [0_u8; 12];
            scales.copy_from_slice(&chunk[4..16]);
            let mut values = [0_u8; Q4_K_BLOCK_SIZE / 2];
            values.copy_from_slice(&chunk[16..]);
            blocks.push(Q4KBlock {
                scale,
                min_scale,
                scales,
                values,
            });
        }
        Ok(QuantizedTensorQ4K::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q5_k(&self, name: &str) -> Result<QuantizedTensorQ5K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q5_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q5_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        if !bytes.len().is_multiple_of(176) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q5_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 176);
        for chunk in bytes.as_chunks::<176>().0 {
            let scale = f16::from_le_bytes(chunk[..2].try_into().unwrap()).to_f32();
            let min_scale = f16::from_le_bytes(chunk[2..4].try_into().unwrap()).to_f32();
            if !scale.is_finite() || !min_scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q5_K tensor {name} contains a non-finite scale"
                )));
            }
            let mut scales = [0_u8; 12];
            scales.copy_from_slice(&chunk[4..16]);
            let mut high_bits = [0_u8; 32];
            high_bits.copy_from_slice(&chunk[16..48]);
            let mut values = [0_u8; 128];
            values.copy_from_slice(&chunk[48..]);
            blocks.push(Q5KBlock {
                scale,
                min_scale,
                scales,
                high_bits,
                values,
            });
        }
        Ok(QuantizedTensorQ5K::from_blocks(info.shape.clone(), blocks)?)
    }

    pub fn tensor_q6_k(&self, name: &str) -> Result<QuantizedTensorQ6K> {
        let info = self.tensor_info(name)?;
        if info.dtype != GgmlType::Q6_K {
            return Err(FormatError::Invalid(format!(
                "GGUF tensor {name} has type {:?}, expected Q6_K",
                info.dtype
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let legacy = self.uses_legacy_quant_block_layout()?;
        if !bytes.len().is_multiple_of(210) {
            return Err(FormatError::Invalid(format!(
                "GGUF Q6_K tensor {name} has an invalid byte length"
            )));
        }
        let mut blocks = Vec::with_capacity(bytes.len() / 210);
        for chunk in bytes.as_chunks::<210>().0 {
            let (scale, scales, low_bits, high_bits) = if legacy {
                (&chunk[..2], &chunk[2..18], &chunk[18..146], &chunk[146..])
            } else {
                (
                    &chunk[208..],
                    &chunk[192..208],
                    &chunk[..128],
                    &chunk[128..192],
                )
            };
            let scale = f16::from_le_bytes(scale.try_into().unwrap()).to_f32();
            if !scale.is_finite() {
                return Err(FormatError::Invalid(format!(
                    "GGUF Q6_K tensor {name} contains a non-finite scale"
                )));
            }
            blocks.push(Q6KBlock {
                scale,
                scales: std::array::from_fn(|index| scales[index] as i8),
                low_bits: low_bits.try_into().unwrap(),
                high_bits: high_bits.try_into().unwrap(),
            });
        }
        Ok(QuantizedTensorQ6K::from_blocks(info.shape.clone(), blocks)?)
    }
}
