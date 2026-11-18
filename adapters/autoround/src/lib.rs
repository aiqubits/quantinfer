//! AutoRound configuration and SafeTensors adapter.
//!
//! AutoRound exports ordinary GPTQ-compatible packed linear tensors. This
//! crate owns the AutoRound metadata contract and delegates the byte-level
//! unpacking to the tested GPTQ importer.

#![forbid(unsafe_code)]

pub use quantinfer_adapter_gptq::ZeroPointLayout;
use quantinfer_adapter_gptq::{dequantize as dequantize_gptq, GptqConfig, GptqTensorSource};
use quantinfer_core::Tensor;
use quantinfer_format::{SafeTensorFile, SafeTensorShards};
use serde_json::Value;

/// AutoRound settings needed for a GPTQ-compatible weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoRoundConfig {
    pub bits: usize,
    pub group_size: usize,
    pub sym: bool,
    pub zero_offset: i32,
}

impl AutoRoundConfig {
    pub fn new(bits: usize, group_size: usize, sym: bool) -> Self {
        Self {
            bits,
            group_size,
            sym,
            zero_offset: 1,
        }
    }

    /// Parse the aliases emitted by AutoRound and Hugging Face quantization
    /// configs (`w_bit`/`bits` and `q_group_size`/`group_size`).
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "AutoRound quantization config must be an object".to_owned())?;
        for key in ["quant_method", "quantization_method"] {
            let Some(value) = object.get(key) else {
                continue;
            };
            let method = value
                .as_str()
                .ok_or_else(|| format!("AutoRound {key} must be a string"))?;
            let method = method.to_ascii_lowercase();
            if !matches!(method.as_str(), "autoround" | "auto_round" | "gptq") {
                return Err(format!(
                    "unsupported AutoRound quantization method {method}"
                ));
            }
        }
        let bits = integer_alias(object, &["bits", "w_bit", "weight_bits"])?
            .ok_or_else(|| "AutoRound config is missing bits/w_bit".to_owned())?;
        let group_size = integer_alias(object, &["group_size", "q_group_size"])?
            .ok_or_else(|| "AutoRound config is missing group_size/q_group_size".to_owned())?;
        let explicit_sym = boolean_alias(object, &["sym", "symmetric"])?;
        let zero_point = boolean_alias(object, &["zero_point"])?;
        let sym = explicit_sym
            .or(zero_point.map(|value| !value))
            .unwrap_or(false);
        let zero_offset = object
            .get("zero_offset")
            .map(|value| {
                let value = value
                    .as_i64()
                    .ok_or_else(|| "AutoRound zero_offset must be an i32 integer".to_owned())?;
                i32::try_from(value).map_err(|_| "AutoRound zero_offset overflows i32".to_owned())
            })
            .transpose()?
            .unwrap_or(1);
        let config = Self {
            bits,
            group_size,
            sym,
            zero_offset,
        };
        config.validate()
    }

    pub fn validate(self) -> Result<Self, String> {
        let gptq = GptqConfig {
            bits: self.bits,
            group_size: self.group_size,
            sym: self.sym,
            zero_offset: self.zero_offset,
        };
        gptq.validate()?;
        Ok(self)
    }

    fn as_gptq(&self) -> GptqConfig {
        GptqConfig {
            bits: self.bits,
            group_size: self.group_size,
            sym: self.sym,
            zero_offset: self.zero_offset,
        }
    }
}

fn integer_alias(
    object: &serde_json::Map<String, Value>,
    names: &[&str],
) -> Result<Option<usize>, String> {
    let mut result = None;
    for name in names {
        if let Some(value) = object.get(*name) {
            let value = value
                .as_u64()
                .ok_or_else(|| format!("AutoRound {name} must be a non-negative integer"))?;
            let value =
                usize::try_from(value).map_err(|_| format!("AutoRound {name} overflows usize"))?;
            if result.is_some_and(|previous| previous != value) {
                return Err(format!("AutoRound {name} conflicts with its alias"));
            }
            result = Some(value);
        }
    }
    Ok(result)
}

fn boolean_alias(
    object: &serde_json::Map<String, Value>,
    names: &[&str],
) -> Result<Option<bool>, String> {
    let mut result = None;
    for name in names {
        if let Some(value) = object.get(*name) {
            let value = value
                .as_bool()
                .ok_or_else(|| format!("AutoRound {name} must be boolean"))?;
            if result.is_some_and(|previous| previous != value) {
                return Err(format!("AutoRound {name} conflicts with its alias"));
            }
            result = Some(value);
        }
    }
    Ok(result)
}

/// Decode an AutoRound matrix whose tensor names follow GPTQ conventions.
pub fn dequantize<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &AutoRoundConfig,
) -> Result<Tensor, String> {
    dequantize_gptq(source, prefix, &config.as_gptq())
}

/// Decode a nonstandard zero-point layout without guessing from a square shape.
pub fn dequantize_with_zero_point_layout<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &AutoRoundConfig,
    layout: ZeroPointLayout,
) -> Result<Tensor, String> {
    quantinfer_adapter_gptq::dequantize_with_zero_point_layout(
        source,
        prefix,
        &config.as_gptq(),
        layout,
    )
}

/// Convenience wrapper for a single SafeTensors file.
pub fn dequantize_file(
    source: &SafeTensorFile,
    prefix: &str,
    config: &AutoRoundConfig,
) -> Result<Tensor, String> {
    dequantize(source, prefix, config)
}

/// Convenience wrapper for an indexed SafeTensors checkpoint.
pub fn dequantize_shards(
    source: &SafeTensorShards,
    prefix: &str,
    config: &AutoRoundConfig,
) -> Result<Tensor, String> {
    dequantize(source, prefix, config)
}

#[cfg(test)]
mod tests;
