//! AutoRound configuration and SafeTensors adapter.
//!
//! AutoRound exports ordinary GPTQ-compatible packed linear tensors. This
//! crate owns the AutoRound metadata contract and delegates the byte-level
//! unpacking to the tested GPTQ importer.

#![forbid(unsafe_code)]

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
        if let Some(method) = object
            .get("quant_method")
            .or_else(|| object.get("quantization_method"))
            .and_then(Value::as_str)
        {
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
        let sym = if let Some(value) = object.get("sym").or_else(|| object.get("symmetric")) {
            value
                .as_bool()
                .ok_or_else(|| "AutoRound sym/symmetric must be boolean".to_owned())?
        } else if let Some(value) = object.get("zero_point") {
            !value
                .as_bool()
                .ok_or_else(|| "AutoRound zero_point must be boolean".to_owned())?
        } else {
            false
        };
        let zero_offset = object
            .get("zero_offset")
            .and_then(Value::as_i64)
            .map(|value| {
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
    for name in names {
        if let Some(value) = object.get(*name) {
            let value = value
                .as_u64()
                .ok_or_else(|| format!("AutoRound {name} must be a non-negative integer"))?;
            return usize::try_from(value)
                .map(Some)
                .map_err(|_| format!("AutoRound {name} overflows usize"));
        }
    }
    Ok(None)
}

/// Decode an AutoRound matrix whose tensor names follow GPTQ conventions.
pub fn dequantize<S: GptqTensorSource + ?Sized>(
    source: &S,
    prefix: &str,
    config: &AutoRoundConfig,
) -> Result<Tensor, String> {
    dequantize_gptq(source, prefix, &config.as_gptq())
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
mod tests {
    use super::{dequantize_file, AutoRoundConfig};
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
    fn parses_autoround_aliases_and_dequantizes_gptq_layout() {
        let config = AutoRoundConfig::from_json(&json!({
            "quant_method": "auto_round",
            "w_bit": 4,
            "q_group_size": 8,
            "zero_point": true
        }))
        .unwrap();
        assert_eq!(config, AutoRoundConfig::new(4, 8, false));
        let file = fixture(&[
            ("layer.qweight", "I32", vec![1, 1], words(&[0x7654_3210])),
            ("layer.qzeros", "I32", vec![1, 1], words(&[0x0000_0000])),
            ("layer.scales", "F32", vec![1, 1], floats(&[2.0])),
        ]);
        let tensor = dequantize_file(&file, "layer", &config).unwrap();
        assert_eq!(
            tensor.as_f32().unwrap(),
            &[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }

    #[test]
    fn rejects_unknown_method_and_invalid_values() {
        assert!(AutoRoundConfig::from_json(&json!({
            "quant_method": "awq", "bits": 4, "group_size": 128
        }))
        .is_err());
        assert!(AutoRoundConfig::from_json(&json!({
            "bits": 3, "group_size": 128
        }))
        .is_err());
    }
}
