//! SafeTensors JSON header validation.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::types::{SafeTensorDType, SafeTensorInfo};
use crate::{FormatError, Result};

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
    Ok((tensors, metadata))
}

/// JSON objects are maps in `serde_json::Value`, so duplicate keys would
/// otherwise be silently replaced by the last entry. SafeTensors metadata and
/// tensor names must be unambiguous across readers.
struct DuplicateFreeJson(Value);

impl<'de> Deserialize<'de> for DuplicateFreeJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct JsonVisitor;

        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = DuplicateFreeJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(DuplicateFreeJson(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateFreeJson(Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateFreeJson(Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("invalid JSON number"))?;
                Ok(DuplicateFreeJson(Value::Number(number)))
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(DuplicateFreeJson(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
                Ok(DuplicateFreeJson(Value::String(value)))
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(DuplicateFreeJson(Value::Null))
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(DuplicateFreeJson(Value::Null))
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<DuplicateFreeJson>()? {
                    values.push(value.0);
                }
                Ok(DuplicateFreeJson(Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    let value = map.next_value::<DuplicateFreeJson>()?;
                    if object.insert(key.clone(), value.0).is_some() {
                        return Err(de::Error::custom(format!(
                            "duplicate JSON object key {key}"
                        )));
                    }
                }
                Ok(DuplicateFreeJson(Value::Object(object)))
            }
        }

        deserializer.deserialize_any(JsonVisitor)
    }
}
