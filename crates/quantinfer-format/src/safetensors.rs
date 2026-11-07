use std::collections::BTreeMap;

use half::{bf16, f16};

use super::{FormatError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    F32,
    F16,
    BF16,
}

impl SafeTensorDType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            _ => Err(FormatError::Invalid(format!(
                "unsupported SafeTensors dtype {value}"
            ))),
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct SafeTensorFile {
    data: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, SafeTensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensorFile {
    pub fn open(data: Vec<u8>) -> Result<Self> {
        if data.len() < 8 {
            return Err(FormatError::Invalid(
                "SafeTensors file is truncated".to_owned(),
            ));
        }
        let header_size =
            usize::try_from(u64::from_le_bytes(data[0..8].try_into().map_err(|_| {
                FormatError::Invalid("invalid SafeTensors header".to_owned())
            })?))
            .map_err(|_| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        let header_start: usize = 8;
        let data_start = header_start
            .checked_add(header_size)
            .ok_or_else(|| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        let header_end = data_start;
        let header = data
            .get(header_start..header_end)
            .ok_or_else(|| FormatError::Invalid("SafeTensors header is truncated".to_owned()))?;
        let value: serde_json::Value = serde_json::from_slice(header)
            .map_err(|error| FormatError::Invalid(format!("invalid SafeTensors JSON: {error}")))?;
        let object = value.as_object().ok_or_else(|| {
            FormatError::Invalid("SafeTensors header must be an object".to_owned())
        })?;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, entry) in object {
            if name == "__metadata__" {
                if let Some(entries) = entry.as_object() {
                    for (key, value) in entries {
                        if let Some(value) = value.as_str() {
                            metadata.insert(key.clone(), value.to_owned());
                        }
                    }
                }
                continue;
            }
            let entry = entry
                .as_object()
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} is not an object")))?;
            let dtype = SafeTensorDType::parse(
                entry
                    .get("dtype")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no dtype")))?,
            )?;
            let shape = entry
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no shape")))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| {
                            FormatError::Invalid(format!("tensor {name} has an invalid shape"))
                        })
                        .and_then(|value| {
                            usize::try_from(value).map_err(|_| {
                                FormatError::Invalid(format!("tensor {name} shape overflows usize"))
                            })
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    FormatError::Invalid(format!("tensor {name} has no data offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(FormatError::Invalid(format!(
                    "tensor {name} must have two data offsets"
                )));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has invalid offsets")))
                .and_then(|value| {
                    usize::try_from(value).map_err(|_| {
                        FormatError::Invalid(format!("tensor {name} start offset overflows usize"))
                    })
                })?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has invalid offsets")))
                .and_then(|value| {
                    usize::try_from(value).map_err(|_| {
                        FormatError::Invalid(format!("tensor {name} end offset overflows usize"))
                    })
                })?;
            if start > end {
                return Err(FormatError::Invalid(format!(
                    "tensor {name} offsets exceed file bounds"
                )));
            }
            let byte_start = data_start.checked_add(start).ok_or_else(|| {
                FormatError::Invalid(format!("tensor {name} start exceeds file bounds"))
            })?;
            let byte_end = data_start.checked_add(end).ok_or_else(|| {
                FormatError::Invalid(format!("tensor {name} end exceeds file bounds"))
            })?;
            if byte_end > data.len() {
                return Err(FormatError::Invalid(format!(
                    "tensor {name} offsets exceed file bounds"
                )));
            }
            let elements = shape.iter().try_fold(1usize, |product, dimension| {
                product.checked_mul(*dimension).ok_or_else(|| {
                    FormatError::Invalid(format!("tensor {name} shape overflows usize"))
                })
            })?;
            let expected_size = elements.checked_mul(dtype.element_size()).ok_or_else(|| {
                FormatError::Invalid(format!("tensor {name} size overflows usize"))
            })?;
            if byte_end - byte_start != expected_size {
                return Err(FormatError::Invalid(format!(
                    "tensor {name} byte length does not match its shape"
                )));
            }
            tensors.insert(
                name.clone(),
                SafeTensorInfo {
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );
        }
        Ok(Self {
            data,
            data_start,
            tensors,
            metadata,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, SafeTensorInfo> {
        &self.tensors
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let start = self
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors start offset overflows usize".to_owned())
            })?;
        let end = self
            .data_start
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors end offset overflows usize".to_owned())
            })?;
        self.data
            .get(start..end)
            .ok_or_else(|| FormatError::Invalid("SafeTensors data is truncated".to_owned()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let bytes = self.tensor_bytes(name)?;
        let element_size = info.dtype.element_size();
        if !bytes.len().is_multiple_of(element_size) {
            return Err(FormatError::Invalid(format!(
                "tensor {name} has a non-aligned byte length"
            )));
        }
        Ok(match info.dtype {
            SafeTensorDType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect(),
            SafeTensorDType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
                .collect(),
            SafeTensorDType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SafeTensorFile;

    #[test]
    fn reads_f32_tensor_and_metadata() {
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        let file = SafeTensorFile::open(bytes).unwrap();
        assert_eq!(file.metadata().get("format"), Some(&"pt".to_owned()));
        assert_eq!(file.tensor_f32("weight").unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn rejects_tensor_bytes_that_do_not_match_shape() {
        let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        assert!(SafeTensorFile::open(bytes).is_err());
    }
}
