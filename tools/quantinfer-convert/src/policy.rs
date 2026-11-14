//! Quantization and mixed-precision policy parsing.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::common::is_quantizable_linear;
use quantinfer_core::QuantizationScheme;
use quantinfer_quant::{MixedPrecisionPlan, NativeQ8_0Provider, QuantizationProvider};

pub(crate) fn q8_provider(use_ggml: bool) -> Result<&'static dyn QuantizationProvider, String> {
    if !use_ggml {
        return Ok(&NativeQ8_0Provider);
    }
    #[cfg(feature = "ggml")]
    {
        Ok(&quantinfer_adapter_ggml::GgmlQ8_0Provider)
    }
    #[cfg(not(feature = "ggml"))]
    {
        Err("--ggml requires a converter built with the ggml feature".to_owned())
    }
}

pub(crate) fn parse_quantization(value: &str) -> Result<QuantizationScheme, String> {
    match value {
        "f32" => Ok(QuantizationScheme::None),
        "q8_0" => Ok(QuantizationScheme::Q8_0),
        "q4_0" => Ok(QuantizationScheme::Q4_0),
        "q4_k" => Ok(QuantizationScheme::Q4_K),
        "q5_k" => Ok(QuantizationScheme::Q5_K),
        "q6_k" => Ok(QuantizationScheme::Q6_K),
        _ => Err(format!(
            "unsupported quantization {value}; use f32, q8_0, q4_0, q4_k, q5_k, or q6_k"
        )),
    }
}

pub(crate) fn load_mixed_precision(path: &Path) -> Result<MixedPrecisionPlan, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let assignments: BTreeMap<String, String> =
        serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "cannot parse mixed precision plan {}: {error}",
                path.display()
            )
        })?;
    let mut plan = MixedPrecisionPlan::default();
    for (name, scheme) in assignments {
        let scheme = parse_quantization(&scheme).map_err(|error| format!("{name}: {error}"))?;
        plan.assign(name, scheme);
    }
    Ok(plan)
}

#[derive(Debug, serde::Deserialize)]
struct MixedPrecisionPolicyFile {
    default: String,
    attention: String,
    feed_forward: String,
    #[serde(default = "default_true")]
    preserve_output: bool,
}

fn default_true() -> bool {
    true
}

pub(crate) struct MixedPrecisionPolicy {
    default: QuantizationScheme,
    attention: QuantizationScheme,
    feed_forward: QuantizationScheme,
    preserve_output: bool,
}

impl MixedPrecisionPolicy {
    pub(crate) fn scheme_for(&self, name: &str) -> Option<QuantizationScheme> {
        if self.preserve_output && matches!(name, "token_embd.weight" | "output.weight") {
            return None;
        }
        if name.ends_with("attn_q.weight")
            || name.ends_with("attn_k.weight")
            || name.ends_with("attn_v.weight")
            || name.ends_with("attn_output.weight")
        {
            Some(self.attention)
        } else if name.ends_with("ffn_gate.weight")
            || name.ends_with("ffn_up.weight")
            || name.ends_with("ffn_down.weight")
        {
            Some(self.feed_forward)
        } else if is_quantizable_linear(name) {
            Some(self.default)
        } else {
            None
        }
    }
}

pub(crate) fn load_mixed_precision_policy(path: &Path) -> Result<MixedPrecisionPolicy, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let policy: MixedPrecisionPolicyFile = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "cannot parse mixed precision policy {}: {error}",
            path.display()
        )
    })?;
    let parse = |name: &str, value: &str| {
        parse_quantization(value).map_err(|error| format!("mixed precision {name}: {error}"))
    };
    Ok(MixedPrecisionPolicy {
        default: parse("default", &policy.default)?,
        attention: parse("attention", &policy.attention)?,
        feed_forward: parse("feed_forward", &policy.feed_forward)?,
        preserve_output: policy.preserve_output,
    })
}
