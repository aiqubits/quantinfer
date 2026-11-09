use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

use quantinfer_core::{DType, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor};
use quantinfer_format::{
    ExecutionVariant, GgufValue, GgufWriter, ModelManifest, SafeTensorFile, SafeTensorShards,
    TensorManifest,
};
use quantinfer_model::Qwen3Config;
use quantinfer_quant::{quantize_q4_0, quantize_q8_0};
use serde_json::Value;

fn main() {
    if let Err(error) = run() {
        eprintln!("quantinfer-convert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print_help();
        return Ok(());
    }
    let config_path = required_path(&arguments, "--config")?;
    let weights_path = required_path(&arguments, "--weights")?;
    let output_path = required_path(&arguments, "--output")?;
    let manifest_path = optional_value(&arguments, "--manifest").map(PathBuf::from);
    let tokenizer_path = optional_value(&arguments, "--tokenizer").map(PathBuf::from);
    let quantization = optional_value(&arguments, "--quantize");
    if let Some(value) = &quantization {
        if value != "q8_0" && value != "q4_0" {
            return Err(format!(
                "unsupported quantization {value}; use q8_0 or q4_0"
            ));
        }
    }

    let config_json = fs::read_to_string(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    let config = Qwen3Config::from_json(&config_json)
        .map_err(|error| format!("cannot parse Qwen3 config: {error}"))?;
    config
        .validate()
        .map_err(|error| format!("invalid Qwen3 config: {error}"))?;
    let weights = WeightSource::open(&weights_path)?;

    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("qwen3".to_owned()),
    );
    writer.insert_metadata(
        "qwen3.context_length",
        GgufValue::U32(metadata_u32(
            config.max_position_embeddings,
            "max_position_embeddings",
        )?),
    );
    writer.insert_metadata(
        "qwen3.embedding_length",
        GgufValue::U32(metadata_u32(config.hidden_size, "hidden_size")?),
    );
    writer.insert_metadata(
        "qwen3.vocab_size",
        GgufValue::U32(metadata_u32(config.vocab_size, "vocab_size")?),
    );
    writer.insert_metadata(
        "qwen3.feed_forward_length",
        GgufValue::U32(metadata_u32(config.intermediate_size, "intermediate_size")?),
    );
    writer.insert_metadata(
        "qwen3.block_count",
        GgufValue::U32(metadata_u32(config.num_hidden_layers, "num_hidden_layers")?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count",
        GgufValue::U32(metadata_u32(
            config.num_attention_heads,
            "num_attention_heads",
        )?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count_kv",
        GgufValue::U32(metadata_u32(
            config.num_key_value_heads,
            "num_key_value_heads",
        )?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_dim",
        GgufValue::U32(metadata_u32(config.effective_head_dim(), "head_dim")?),
    );
    writer.insert_metadata("qwen3.rope.freq_base", GgufValue::F32(config.rope_theta));
    writer.insert_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        GgufValue::F32(config.rms_norm_eps),
    );
    writer.insert_metadata(
        "qwen3.tie_word_embeddings",
        GgufValue::Bool(config.tie_word_embeddings),
    );
    writer.insert_metadata(
        "qwen3.architectures",
        GgufValue::Array(
            config
                .architectures
                .iter()
                .cloned()
                .map(GgufValue::String)
                .collect(),
        ),
    );
    if let Some(tokenizer_path) = tokenizer_path {
        let tokenizer = load_tokenizer_metadata(&tokenizer_path, &config_json)?;
        if tokenizer.tokens.len() > config.vocab_size {
            return Err(format!(
                "tokenizer vocabulary has {} entries, exceeding model vocabulary {}",
                tokenizer.tokens.len(),
                config.vocab_size
            ));
        }
        let mut tokens = tokenizer.tokens;
        while tokens.len() < config.vocab_size {
            let token_id = tokens.len();
            tokens.push(format!("<|reserved_{token_id}|>"));
        }
        writer.insert_metadata(
            "tokenizer.ggml.tokens",
            GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        if !tokenizer.merges.is_empty() {
            writer.insert_metadata(
                "tokenizer.ggml.merges",
                GgufValue::Array(
                    tokenizer
                        .merges
                        .into_iter()
                        .map(GgufValue::String)
                        .collect(),
                ),
            );
        }
        writer.insert_metadata("tokenizer.ggml.model", GgufValue::String("gpt2".to_owned()));
        if let Some(token_id) = tokenizer.bos_token_id {
            writer.insert_metadata(
                "tokenizer.ggml.bos_token_id",
                GgufValue::U32(metadata_u32(token_id, "bos_token_id")?),
            );
        }
        if let Some(token_id) = tokenizer.eos_token_id {
            writer.insert_metadata(
                "tokenizer.ggml.eos_token_id",
                GgufValue::U32(metadata_u32(token_id, "eos_token_id")?),
            );
        }
    }
    let mut written_names = BTreeSet::new();
    let mut manifest_tensors = Vec::new();
    for (name, shape) in weights.tensors()? {
        let values = weights
            .tensor_f32(&name)
            .map_err(|error| format!("cannot decode tensor {name}: {error}"))?;
        let canonical_name = canonical_tensor_name(&name);
        let output_name = canonical_name.as_deref().unwrap_or(&name);
        if !written_names.insert(output_name.to_owned()) {
            return Err(format!(
                "multiple SafeTensors entries map to GGUF tensor {output_name}"
            ));
        }
        let (shape, values) = canonicalize_tensor_layout(output_name, &shape, values)?;
        let use_q8 = quantization.as_deref() == Some("q8_0")
            && is_quantizable_linear(output_name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q8_0_BLOCK_SIZE);
        let use_q4 = quantization.as_deref() == Some("q4_0")
            && is_quantizable_linear(output_name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q4_0_BLOCK_SIZE);
        let descriptor = if use_q8 {
            TensorStorageDescriptor::quantized(
                DType::Q8_0,
                QuantizationScheme::Q8_0,
                Shape::from(shape.clone()),
                quantinfer_core::Q8_0_BLOCK_SIZE,
            )
        } else if use_q4 {
            TensorStorageDescriptor::quantized(
                DType::Q4_0,
                QuantizationScheme::Q4_0,
                Shape::from(shape.clone()),
                quantinfer_core::Q4_0_BLOCK_SIZE,
            )
        } else {
            TensorStorageDescriptor::floating(DType::F32, Shape::from(shape.clone()))
        }
        .map_err(|error| format!("cannot describe tensor {name}: {error}"))?;
        manifest_tensors.push(TensorManifest {
            name: output_name.to_owned(),
            descriptor,
            checksum_sha256: None,
        });
        if use_q8 {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q8_0(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q8_0_tensor(output_name, &quantized);
        } else if use_q4 {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q4_0(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q4_0_tensor(output_name, &quantized);
        } else {
            writer
                .add_f32_tensor(output_name, shape, &values)
                .map_err(|error| format!("cannot add tensor {name}: {error}"))?;
        }
    }
    validate_required_tensors(&config, &written_names)?;
    let bytes = writer
        .finish()
        .map_err(|error| format!("cannot write GGUF: {error}"))?;
    fs::write(&output_path, &bytes)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    if let Some(manifest_path) = manifest_path {
        let mut manifest = ModelManifest::new("qwen3", "qwen3");
        manifest.source_format = "safetensors-to-gguf".to_owned();
        manifest.quantization = match quantization.as_deref() {
            Some("q8_0") => QuantizationScheme::Q8_0,
            Some("q4_0") => QuantizationScheme::Q4_0,
            _ => QuantizationScheme::None,
        };
        manifest.tensors = manifest_tensors;
        manifest.execution_variants.push(ExecutionVariant {
            name: "cpu-scalar".to_owned(),
            quantization: manifest.quantization,
            dtype: "f32-accumulation".to_owned(),
            runtime_provider: "cpu-scalar".to_owned(),
            kernel_provider: "cpu-scalar".to_owned(),
        });
        let output_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.gguf");
        manifest.insert_checksum(output_name, &bytes);
        let json = manifest
            .to_json()
            .map_err(|error| format!("cannot serialize model manifest: {error}"))?;
        fs::write(&manifest_path, json)
            .map_err(|error| format!("cannot write {}: {error}", manifest_path.display()))?;
    }
    println!(
        "converted {} tensors to {}",
        weights.tensor_count(),
        output_path.display()
    );
    Ok(())
}

enum WeightSource {
    Single(SafeTensorFile),
    Sharded(SafeTensorShards),
}

impl WeightSource {
    fn open(path: &PathBuf) -> Result<Self, String> {
        if path.to_string_lossy().ends_with(".index.json") {
            SafeTensorShards::open(path)
                .map(Self::Sharded)
                .map_err(|error| format!("cannot parse SafeTensors index: {error}"))
        } else {
            SafeTensorFile::open_path(path)
                .map(Self::Single)
                .map_err(|error| format!("cannot parse SafeTensors: {error}"))
        }
    }

    fn tensors(&self) -> Result<Vec<(String, Vec<usize>)>, String> {
        match self {
            Self::Single(file) => Ok(file
                .tensors()
                .iter()
                .map(|(name, info)| (name.clone(), info.shape.clone()))
                .collect()),
            Self::Sharded(files) => Ok(files
                .tensors()
                .map(|(name, info)| (name.to_owned(), info.shape.clone()))
                .collect()),
        }
    }

    fn tensor_f32(&self, name: &str) -> quantinfer_format::Result<Vec<f32>> {
        match self {
            Self::Single(file) => file.tensor_f32(name),
            Self::Sharded(files) => files.tensor_f32(name),
        }
    }

    fn tensor_count(&self) -> usize {
        match self {
            Self::Single(file) => file.tensors().len(),
            Self::Sharded(files) => files.tensors().count(),
        }
    }
}

fn validate_required_tensors(
    config: &Qwen3Config,
    written_names: &BTreeSet<String>,
) -> Result<(), String> {
    let mut required = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
    ];
    if !config.tie_word_embeddings {
        required.push("output.weight".to_owned());
    }
    for layer in 0..config.num_hidden_layers {
        for suffix in [
            "attn_norm",
            "attn_q",
            "attn_q_norm",
            "attn_k",
            "attn_k_norm",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            required.push(format!("blk.{layer}.{suffix}.weight"));
        }
    }
    let missing: Vec<_> = required
        .into_iter()
        .filter(|name| !written_names.contains(name))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "converted checkpoint is missing required Qwen3 tensors: {}",
            missing.join(", ")
        ))
    }
}

fn required_path(arguments: &[String], flag: &str) -> Result<PathBuf, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {flag}; use --help for usage"))
}

fn optional_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn metadata_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} does not fit in GGUF u32 metadata"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizerMetadata {
    tokens: Vec<String>,
    merges: Vec<String>,
    bos_token_id: Option<usize>,
    eos_token_id: Option<usize>,
}

fn load_tokenizer_metadata(
    tokenizer_path: &PathBuf,
    config_json: &str,
) -> Result<TokenizerMetadata, String> {
    let tokenizer_json = fs::read_to_string(tokenizer_path)
        .map_err(|error| format!("cannot read {}: {error}", tokenizer_path.display()))?;
    let root: Value = serde_json::from_str(&tokenizer_json)
        .map_err(|error| format!("cannot parse {}: {error}", tokenizer_path.display()))?;
    let vocab = root
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(Value::as_object)
        .ok_or_else(|| "tokenizer.json is missing model.vocab".to_owned())?;
    let maximum_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .ok_or_else(|| "tokenizer.json model.vocab is empty".to_owned())?;
    let maximum_id = usize::try_from(maximum_id)
        .map_err(|_| "tokenizer vocabulary id overflows usize".to_owned())?;
    let token_count = maximum_id
        .checked_add(1)
        .ok_or_else(|| "tokenizer vocabulary size overflows usize".to_owned())?;
    let mut tokens = vec![None; token_count];
    for (token, value) in vocab {
        let token_id = value
            .as_u64()
            .ok_or_else(|| format!("tokenizer id for {token:?} is not an integer"))?;
        let token_id = usize::try_from(token_id)
            .map_err(|_| format!("tokenizer id for {token:?} overflows usize"))?;
        tokens[token_id] = Some(token.clone());
    }
    if let Some(added_tokens) = root.get("added_tokens").and_then(Value::as_array) {
        for added_token in added_tokens {
            let Some(token) = added_token.get("content").and_then(Value::as_str) else {
                continue;
            };
            let Some(token_id) = added_token.get("id").and_then(Value::as_u64) else {
                continue;
            };
            let token_id = usize::try_from(token_id)
                .map_err(|_| format!("added tokenizer id for {token:?} overflows usize"))?;
            if token_id >= tokens.len() {
                tokens.resize(token_id + 1, None);
            }
            tokens[token_id] = Some(token.to_owned());
        }
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(token_id, token)| {
            token.ok_or_else(|| format!("tokenizer vocabulary is missing id {token_id}"))
        })
        .collect::<Result<_, _>>()?;
    let merges = root
        .get("model")
        .and_then(|model| model.get("merges"))
        .and_then(Value::as_array)
        .map(|merges| {
            merges
                .iter()
                .map(|merge| {
                    if let Some(merge) = merge.as_str() {
                        Ok(merge.to_owned())
                    } else if let Some(parts) = merge.as_array() {
                        if parts.len() != 2 {
                            return Err(
                                "tokenizer merge arrays must contain two strings".to_owned()
                            );
                        }
                        let left = parts[0].as_str().ok_or_else(|| {
                            "tokenizer merge left side is not a string".to_owned()
                        })?;
                        let right = parts[1].as_str().ok_or_else(|| {
                            "tokenizer merge right side is not a string".to_owned()
                        })?;
                        Ok(format!("{left} {right}"))
                    } else {
                        Err("tokenizer merge must be a string or two-item array".to_owned())
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let config: Value = serde_json::from_str(config_json)
        .map_err(|error| format!("cannot parse Qwen3 config for tokenizer ids: {error}"))?;
    Ok(TokenizerMetadata {
        tokens,
        merges,
        bos_token_id: token_id_from_value(config.get("bos_token_id")),
        eos_token_id: token_id_from_value(config.get("eos_token_id")),
    })
}

fn token_id_from_value(value: Option<&Value>) -> Option<usize> {
    value.and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_array()?.first()?.as_u64())
            .and_then(|id| usize::try_from(id).ok())
    })
}

fn canonical_tensor_name(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.as_slice() {
        ["model", "embed_tokens", "weight"] | ["model", "token_embeddings", "weight"] => {
            Some("token_embd.weight".to_owned())
        }
        ["model", "norm", "weight"] => Some("output_norm.weight".to_owned()),
        ["lm_head", "weight"] => Some("output.weight".to_owned()),
        ["model", "layers", index, "input_layernorm", "weight"] => {
            Some(format!("blk.{index}.attn_norm.weight"))
        }
        ["model", "layers", index, "self_attn", projection, "weight"] => {
            let suffix = match *projection {
                "q_proj" => "attn_q",
                "q_norm" => "attn_q_norm",
                "k_proj" => "attn_k",
                "k_norm" => "attn_k_norm",
                "v_proj" => "attn_v",
                "o_proj" => "attn_output",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        ["model", "layers", index, "post_attention_layernorm", "weight"] => {
            Some(format!("blk.{index}.ffn_norm.weight"))
        }
        ["model", "layers", index, "mlp", projection, "weight"] => {
            let suffix = match *projection {
                "gate_proj" => "ffn_gate",
                "up_proj" => "ffn_up",
                "down_proj" => "ffn_down",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        _ => None,
    }
}

fn canonicalize_tensor_layout(
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if !is_matrix_weight(name) {
        return Ok((shape.to_vec(), values));
    }
    if shape.len() != 2 {
        return Err(format!("linear tensor {name} must be rank 2"));
    }
    let rows = shape[0];
    let columns = shape[1];
    if values.len() != rows * columns {
        return Err(format!("linear tensor {name} shape does not match data"));
    }
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }
    Ok((vec![columns, rows], transposed))
}

fn is_matrix_weight(name: &str) -> bool {
    name == "output.weight"
        || [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn is_quantizable_linear(name: &str) -> bool {
    [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

fn print_help() {
    println!(
        "Usage: quantinfer-convert --config config.json --weights model.safetensors --output model.gguf [--tokenizer tokenizer.json] [--manifest manifest.json] [--quantize q8_0|q4_0]"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_tensor_name, canonicalize_tensor_layout, is_matrix_weight, is_quantizable_linear,
        load_tokenizer_metadata,
    };

    #[test]
    fn maps_qwen3_huggingface_names_to_gguf_names() {
        assert_eq!(
            canonical_tensor_name("model.layers.2.self_attn.q_proj.weight"),
            Some("blk.2.attn_q.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.layers.2.mlp.down_proj.weight"),
            Some("blk.2.ffn_down.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.embed_tokens.weight"),
            Some("token_embd.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.norm.weight"),
            Some("output_norm.weight".to_owned())
        );
    }

    #[test]
    fn transposes_runtime_linear_layout() {
        let (shape, values) = canonicalize_tensor_layout(
            "blk.0.attn_q.weight",
            &[2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(is_matrix_weight("blk.0.attn_q.weight"));
        assert!(is_quantizable_linear("blk.0.attn_q.weight"));
        assert!(!is_matrix_weight("blk.0.attn_q_norm.weight"));
    }

    #[test]
    fn reads_huggingface_tokenizer_vocab_and_merges() {
        let directory = std::env::temp_dir();
        let path = directory.join(format!("quantinfer-tokenizer-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{
                "model": {
                    "vocab": {"h": 0, "i": 1, "hi": 2},
                    "merges": [["h", "i"]]
                },
                "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}]
            }"#,
        )
        .unwrap();
        let metadata =
            load_tokenizer_metadata(&path, r#"{"bos_token_id": 0, "eos_token_id": 3}"#).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(metadata.tokens, vec!["h", "i", "hi", "<|im_end|>"]);
        assert_eq!(metadata.merges, vec!["h i"]);
        assert_eq!(metadata.eos_token_id, Some(3));
    }
}
