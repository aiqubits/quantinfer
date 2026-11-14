//! Load Hugging Face PEFT LoRA factors for QuantInfer Dense models.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;

use quantinfer_core::Tensor;
use quantinfer_format::{read_file_nofollow, SafeTensorFile};
use quantinfer_model::{ModelSpec, TensorRole};
use quantinfer_runtime::{
    canonical_dense_tensor_name, dense_storage_name, LoraAdapter, LoraProjection,
};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct PeftConfig {
    peft_type: String,
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    rank_pattern: BTreeMap<String, usize>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f32>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl PeftConfig {
    fn validate(&self) -> Result<(), String> {
        if self.peft_type != "LORA" {
            return Err(format!("unsupported PEFT type {}", self.peft_type));
        }
        if self.r == 0 || self.rank_pattern.values().any(|&rank| rank == 0) {
            return Err("PEFT LoRA rank must be greater than zero".to_owned());
        }
        if self
            .extra
            .get("bias")
            .is_some_and(|value| !value.is_null() && value.as_str() != Some("none"))
        {
            return Err("PEFT LoRA bias must be none".to_owned());
        }
        if self.extra.get("init_lora_weights").is_some_and(|value| {
            !(value.is_null() || value.is_boolean() || value.as_str() == Some("gaussian"))
        }) {
            return Err("unsupported PEFT LoRA initializer".to_owned());
        }
        for field in [
            "alora_invocation_tokens",
            "bd_lora_config",
            "corda_config",
            "ensure_weight_tying",
            "eva_config",
            "fan_in_fan_out",
            "layer_replication",
            "loftq_config",
            "lora_bias",
            "modules_to_save",
            "target_modules_to_tie",
            "target_parameters",
            "trainable_token_indices",
            "use_dora",
            "use_qalora",
        ] {
            if self.extra.get(field).is_some_and(|value| match value {
                Value::Null | Value::Bool(false) => false,
                Value::Array(items) => !items.is_empty(),
                Value::Object(entries) => !entries.is_empty(),
                _ => true,
            }) {
                return Err(format!("unsupported PEFT LoRA option {field}"));
            }
        }
        Ok(())
    }
}

/// Load linear A/B factors from `adapter_config.json` and `adapter_model.safetensors`.
/// The caller supplies the specification of the base model used to train the adapter.
pub fn load_peft(directory: impl AsRef<Path>, spec: &ModelSpec) -> Result<LoraAdapter, String> {
    let directory = directory.as_ref();
    let config_path = directory.join("adapter_config.json");
    let config: PeftConfig = serde_json::from_slice(
        &read_file_nofollow(&config_path)
            .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("cannot parse {}: {error}", config_path.display()))?;
    config.validate()?;
    let ranks = compile_patterns(&config.rank_pattern)?;
    let alphas = compile_patterns(&config.alpha_pattern)?;
    let weights_path = directory.join("adapter_model.safetensors");
    let weights = SafeTensorFile::open_path(&weights_path)
        .map_err(|error| format!("cannot load {}: {error}", weights_path.display()))?;
    let mut pairs = BTreeMap::<&str, [Option<&str>; 2]>::new();
    for name in weights.tensors().keys() {
        let (module, factor) = name
            .rsplit_once(".lora_")
            .ok_or_else(|| format!("unsupported PEFT tensor {name}"))?;
        let index = match factor {
            "A.weight" | "A.default.weight" => 0,
            "B.weight" | "B.default.weight" => 1,
            _ => return Err(format!("unsupported PEFT tensor {name}")),
        };
        let module = module.strip_prefix("base_model.model.").unwrap_or(module);
        if pairs.entry(module).or_default()[index]
            .replace(name)
            .is_some()
        {
            return Err(format!("duplicate PEFT LoRA factor for {module}"));
        }
    }
    if pairs.is_empty() {
        return Err("PEFT adapter contains no LoRA factors".to_owned());
    }
    let mut adapter = LoraAdapter::default();
    for (module, pair) in pairs {
        let [Some(a), Some(b)] = pair else {
            return Err(format!(
                "PEFT LoRA requires both A and B factors for {module}"
            ));
        };
        let rank = module_value(&ranks, module, config.r)?;
        let alpha = module_value(&alphas, module, config.lora_alpha)?;
        let (role, layer, [input, output]) = projection_target(spec, module)?;
        let divisor = if config.use_rslora {
            (rank as f32).sqrt()
        } else {
            rank as f32
        };
        let a = factor_tensor(&weights, a, [rank, input])?;
        let b = factor_tensor(&weights, b, [output, rank])?;
        let projection = LoraProjection::from_peft(&a, &b, alpha / divisor)
            .map_err(|error| format!("invalid PEFT LoRA factors for {module}: {error}"))?;
        adapter
            .insert(layer, role, projection)
            .map_err(|error| format!("cannot target {module}: {error}"))?;
    }
    Ok(adapter)
}

fn projection_target(
    spec: &ModelSpec,
    module: &str,
) -> Result<(TensorRole, Option<usize>, [usize; 2]), String> {
    let name = canonical_dense_tensor_name(&format!("{module}.weight"))
        .ok_or_else(|| format!("unsupported PEFT LoRA target {module}"))?;
    if name == "output.weight" {
        return Ok((
            TensorRole::LmHead,
            None,
            [spec.hidden_size, spec.vocabulary.size],
        ));
    }
    let (role, layer, tensor) = spec
        .tensor_map
        .iter()
        .find_map(|tensor| {
            (dense_storage_name(tensor.role, tensor.layer) == name).then_some((
                tensor.role,
                tensor.layer,
                tensor,
            ))
        })
        .ok_or_else(|| format!("PEFT LoRA target {module} is absent from the model"))?;
    let [input, output] = *tensor.logical_shape.dims() else {
        return Err(format!("PEFT LoRA target {module} must be a matrix"));
    };
    Ok((role, layer, [input, output]))
}

fn factor_tensor(
    weights: &SafeTensorFile,
    name: &str,
    shape: [usize; 2],
) -> Result<Tensor, String> {
    let actual = &weights.tensors()[name].shape;
    if actual.as_slice() != shape {
        return Err(format!(
            "PEFT tensor {name} has shape {actual:?}, expected {shape:?}"
        ));
    }
    let data = weights
        .tensor_f32(name)
        .map_err(|error| error.to_string())?;
    Tensor::from_f32(shape.to_vec(), data).map_err(|error| error.to_string())
}

fn compile_patterns<T: Copy>(patterns: &BTreeMap<String, T>) -> Result<Vec<(Regex, T)>, String> {
    patterns
        .iter()
        .map(|(pattern, &value)| {
            Regex::new(&format!(r"^(?:.*\.)?(?:{pattern})$"))
                .map(|regex| (regex, value))
                .map_err(|error| format!("unsupported PEFT pattern {pattern:?}: {error}"))
        })
        .collect()
}

fn module_value<T: Copy>(patterns: &[(Regex, T)], module: &str, default: T) -> Result<T, String> {
    let mut matches = patterns.iter().filter(|(regex, _)| regex.is_match(module));
    let first = matches.next().map(|(_, value)| *value);
    if matches.next().is_some() {
        return Err(format!("overlapping PEFT LoRA patterns for {module}"));
    }
    Ok(first.unwrap_or(default))
}
