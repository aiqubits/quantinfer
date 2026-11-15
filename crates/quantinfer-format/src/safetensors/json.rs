//! SafeTensors JSON header validation.

use serde_json::Value;
use std::collections::BTreeMap;

use super::types::{SafeTensorDType, SafeTensorInfo};
use crate::{DuplicateFreeJson, FormatError, Result};

pub(crate) fn parse_header(
    header: &[u8],
    data_start: usize,
    file_len: usize,
) -> Result<(BTreeMap<String, SafeTensorInfo>, BTreeMap<String, String>)> {
    let value: Value = serde_json::from_slice::<DuplicateFreeJson>(header)
        .map(|value| value.0)
        .map_err(|error| FormatError::Invalid(format!("invalid SafeTensors JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| FormatError::Invalid("SafeTensors header must be an object".to_owned()))?;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            let entries = entry.as_object().ok_or_else(|| {
                FormatError::Invalid("SafeTensors __metadata__ must be an object".to_owned())
            })?;
            for (key, value) in entries {
                let value = value.as_str().ok_or_else(|| {
                    FormatError::Invalid(format!(
                        "SafeTensors metadata value {key} must be a string"
                    ))
                })?;
                metadata.insert(key.clone(), value.to_owned());
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
        if shape.contains(&0) {
            return Err(FormatError::Invalid(format!(
                "tensor {name} dimensions must be greater than zero"
            )));
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no data offsets")))?;
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
        if byte_end > file_len {
            return Err(FormatError::Invalid(format!(
                "tensor {name} offsets exceed file bounds"
            )));
        }
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} shape overflows usize")))
        })?;
        let expected_size = elements
            .checked_mul(dtype.element_size())
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} size overflows usize")))?;
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
    let mut ranges: Vec<(&str, (usize, usize))> = tensors
        .iter()
        .map(|(name, info)| (name.as_str(), info.data_offsets))
        .collect();
    ranges.sort_unstable_by_key(|(_, (start, _))| *start);
    if ranges.windows(2).any(|pair| {
        let previous_end = (pair[0].1).1;
        let next_start = (pair[1].1).0;
        next_start < previous_end
    }) {
        return Err(FormatError::Invalid(
            "SafeTensors tensor data ranges overlap".to_owned(),
        ));
    }
    Ok((tensors, metadata))
}
