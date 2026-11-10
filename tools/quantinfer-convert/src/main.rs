use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

use quantinfer_core::{DType, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor};
use quantinfer_format::{
    ExecutionVariant, GgufValue, GgufWriter, ModelManifest, SafeTensorFile, SafeTensorShards,
    TensorManifest,
};
use quantinfer_model::{LlamaAdapter, LlamaConfig, ModelAdapter, Qwen3Adapter, Qwen3Config};
use quantinfer_quant::{
    quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, NativeQ8_0Provider,
    QuantizationProvider, QuantizedData,
};
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
    let requested_architecture = optional_value(&arguments, "--architecture");
    let quantization = optional_value(&arguments, "--quantize");
    let use_ggml = arguments.iter().any(|argument| argument == "--ggml");
    if use_ggml && quantization.as_deref() != Some("q8_0") {
        return Err("--ggml requires --quantize q8_0".to_owned());
    }
    let q8_provider = q8_provider(use_ggml)?;
    if let Some(value) = &quantization {
        if value != "q8_0"
            && value != "q4_0"
            && value != "q4_k"
            && value != "q5_k"
            && value != "q6_k"
        {
            return Err(format!(
                "unsupported quantization {value}; use q8_0, q4_0, q4_k, q5_k, or q6_k"
            ));
        }
    }

    let config_json = fs::read_to_string(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    let model = ConversionModel::from_config(&config_json, requested_architecture.as_deref())?;
    let weights = WeightSource::open(&weights_path)?;

    let mut writer = GgufWriter::new();
    model.write_metadata(&mut writer)?;
    if use_ggml {
        writer.insert_metadata(
            "quantinfer.quantization.provider",
            GgufValue::String("ggml".to_owned()),
        );
    }
    if let Some(tokenizer_path) = tokenizer_path {
        let tokenizer = load_tokenizer_metadata(&tokenizer_path, &config_json)?;
        if tokenizer.tokens.len() > model.vocab_size() {
            return Err(format!(
                "tokenizer vocabulary has {} entries, exceeding model vocabulary {}",
                tokenizer.tokens.len(),
                model.vocab_size()
            ));
        }
        let mut tokens = tokenizer.tokens;
        while tokens.len() < model.vocab_size() {
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
        let canonical_name = model.canonical_tensor_name(&name);
        let output_name = canonical_name.as_deref().unwrap_or(&name);
        if !written_names.insert(output_name.to_owned()) {
            return Err(format!(
                "multiple SafeTensors entries map to GGUF tensor {output_name}"
            ));
        }
        let (shape, values) = model.canonicalize_tensor_layout(output_name, &shape, values)?;
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
        let use_q4_k = quantization.as_deref() == Some("q4_k")
            && is_quantizable_linear(output_name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q4_K_BLOCK_SIZE);
        let use_q5_k = quantization.as_deref() == Some("q5_k")
            && is_quantizable_linear(output_name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q4_K_BLOCK_SIZE);
        let use_q6_k = quantization.as_deref() == Some("q6_k")
            && is_quantizable_linear(output_name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q4_K_BLOCK_SIZE);
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
        } else if use_q4_k {
            TensorStorageDescriptor::quantized(
                DType::Q4_K,
                QuantizationScheme::Q4_K,
                Shape::from(shape.clone()),
                quantinfer_core::Q4_K_BLOCK_SIZE,
            )
        } else if use_q5_k {
            TensorStorageDescriptor::quantized(
                DType::Q5_K,
                QuantizationScheme::Q5_K,
                Shape::from(shape.clone()),
                quantinfer_core::Q4_K_BLOCK_SIZE,
            )
        } else if use_q6_k {
            TensorStorageDescriptor::quantized(
                DType::Q6_K,
                QuantizationScheme::Q6_K,
                Shape::from(shape.clone()),
                quantinfer_core::Q4_K_BLOCK_SIZE,
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
            let quantized = q8_provider
                .quantize(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            let QuantizedData::Q8_0(quantized) = quantized else {
                return Err(format!(
                    "Q8_0 provider returned a different scheme for {name}"
                ));
            };
            writer.add_q8_0_tensor(output_name, &quantized);
        } else if use_q4 {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q4_0(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q4_0_tensor(output_name, &quantized);
        } else if use_q4_k {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q4_k(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q4_k_tensor(output_name, &quantized);
        } else if use_q5_k {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q5_k(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q5_k_tensor(output_name, &quantized);
        } else if use_q6_k {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q6_k(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q6_k_tensor(output_name, &quantized);
        } else {
            writer
                .add_f32_tensor(output_name, shape, &values)
                .map_err(|error| format!("cannot add tensor {name}: {error}"))?;
        }
    }
    model.validate_required_tensors(&written_names)?;
    let bytes = writer
        .finish()
        .map_err(|error| format!("cannot write GGUF: {error}"))?;
    fs::write(&output_path, &bytes)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    if let Some(manifest_path) = manifest_path {
        let mut manifest = ModelManifest::new(model.architecture(), model.architecture());
        manifest.source_format = "safetensors-to-gguf".to_owned();
        manifest.quantization = match quantization.as_deref() {
            Some("q8_0") => QuantizationScheme::Q8_0,
            Some("q4_0") => QuantizationScheme::Q4_0,
            Some("q4_k") => QuantizationScheme::Q4_K,
            Some("q5_k") => QuantizationScheme::Q5_K,
            Some("q6_k") => QuantizationScheme::Q6_K,
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
        if let Some(parent) = manifest_path.parent() {
            fs::write(parent.join("checksums.sha256"), manifest.checksums_sha256())
                .map_err(|error| format!("cannot write checksums.sha256: {error}"))?;
        }
    }
    println!(
        "converted {} tensors to {}",
        weights.tensor_count(),
        output_path.display()
    );
    Ok(())
}

fn q8_provider(use_ggml: bool) -> Result<&'static dyn QuantizationProvider, String> {
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

enum WeightSource {
    Single(SafeTensorFile),
    Sharded(SafeTensorShards),
}

enum ConversionModel {
    Qwen3 {
        config: Qwen3Config,
        adapter: Qwen3Adapter,
    },
    Llama {
        config: LlamaConfig,
        adapter: LlamaAdapter,
    },
}

impl ConversionModel {
    fn from_config(
        config_json: &str,
        requested_architecture: Option<&str>,
    ) -> Result<Self, String> {
        let root: Value = serde_json::from_str(config_json)
            .map_err(|error| format!("cannot parse model config: {error}"))?;
        let architecture = requested_architecture
            .or_else(|| root.get("model_type").and_then(Value::as_str))
            .or_else(|| {
                root.get("architectures")
                    .and_then(Value::as_array)
                    .and_then(|architectures| {
                        architectures
                            .iter()
                            .filter_map(Value::as_str)
                            .find(|value| {
                                let value = value.to_ascii_lowercase();
                                value.contains("llama") || value.contains("qwen3")
                            })
                    })
            })
            .unwrap_or("qwen3")
            .to_ascii_lowercase();
        match architecture.as_str() {
            value if value == "qwen3" || value == "qwen" || value.contains("qwen3") => {
                let config = Qwen3Config::from_json(config_json)
                    .map_err(|error| format!("cannot parse Qwen3 config: {error}"))?;
                config
                    .validate()
                    .map_err(|error| format!("invalid Qwen3 config: {error}"))?;
                let adapter = Qwen3Adapter::new(config.clone(), "qwen3")
                    .map_err(|error| format!("cannot build Qwen3 model adapter: {error}"))?;
                Ok(Self::Qwen3 { config, adapter })
            }
            value if value == "llama" || value == "llama2" || value.contains("llama") => {
                let config = LlamaConfig::from_json(config_json)
                    .map_err(|error| format!("cannot parse Llama config: {error}"))?;
                config
                    .validate()
                    .map_err(|error| format!("invalid Llama config: {error}"))?;
                let adapter = LlamaAdapter::new(config.clone(), "llama")
                    .map_err(|error| format!("cannot build Llama model adapter: {error}"))?;
                Ok(Self::Llama { config, adapter })
            }
            other => Err(format!(
                "unsupported model architecture {other}; use qwen3 or llama"
            )),
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            Self::Qwen3 { .. } => "qwen3",
            Self::Llama { .. } => "llama",
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen3 { config, .. } => config.vocab_size,
            Self::Llama { config, .. } => config.vocab_size,
        }
    }

    fn canonical_tensor_name(&self, source_name: &str) -> Option<String> {
        match self {
            Self::Qwen3 { adapter, .. } => adapter.canonical_tensor_name(source_name),
            Self::Llama { adapter, .. } => adapter.canonical_tensor_name(source_name),
        }
    }

    fn canonicalize_tensor_layout(
        &self,
        canonical_name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        match self {
            Self::Qwen3 { adapter, .. } => {
                adapter.canonicalize_tensor_layout(canonical_name, shape, values)
            }
            Self::Llama { adapter, .. } => {
                adapter.canonicalize_tensor_layout(canonical_name, shape, values)
            }
        }
    }

    fn write_metadata(&self, writer: &mut GgufWriter) -> Result<(), String> {
        let (
            prefix,
            vocab_size,
            hidden_size,
            intermediate_size,
            layers,
            heads,
            kv_heads,
            head_dim,
            rope_theta,
            rms_norm_eps,
            tie_word_embeddings,
            architectures,
        ) = match self {
            Self::Qwen3 { config, .. } => (
                "qwen3",
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.effective_head_dim(),
                config.rope_theta,
                config.rms_norm_eps,
                config.tie_word_embeddings,
                &config.architectures,
            ),
            Self::Llama { config, .. } => (
                "llama",
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                config.num_attention_heads,
                config.num_key_value_heads(),
                config.effective_head_dim(),
                config.rope_theta,
                config.rms_norm_eps,
                config.tie_word_embeddings,
                &config.architectures,
            ),
        };
        writer.insert_metadata("general.architecture", GgufValue::String(prefix.to_owned()));
        for (key, value) in [
            (
                "context_length",
                match self {
                    Self::Qwen3 { config, .. } => config.max_position_embeddings,
                    Self::Llama { config, .. } => config.max_position_embeddings,
                },
            ),
            ("embedding_length", hidden_size),
            ("vocab_size", vocab_size),
            ("feed_forward_length", intermediate_size),
            ("block_count", layers),
            ("attention.head_count", heads),
            ("attention.head_count_kv", kv_heads),
            ("attention.head_dim", head_dim),
        ] {
            writer.insert_metadata(
                format!("{prefix}.{key}"),
                GgufValue::U32(metadata_u32(value, key)?),
            );
        }
        writer.insert_metadata(
            format!("{prefix}.rope.freq_base"),
            GgufValue::F32(rope_theta),
        );
        writer.insert_metadata(
            format!("{prefix}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(rms_norm_eps),
        );
        writer.insert_metadata(
            format!("{prefix}.tie_word_embeddings"),
            GgufValue::Bool(tie_word_embeddings),
        );
        writer.insert_metadata(
            format!("{prefix}.architectures"),
            GgufValue::Array(
                architectures
                    .iter()
                    .cloned()
                    .map(GgufValue::String)
                    .collect(),
            ),
        );
        Ok(())
    }

    fn validate_required_tensors(&self, written_names: &BTreeSet<String>) -> Result<(), String> {
        let (tie_word_embeddings, layers, qk_norm) = match self {
            Self::Qwen3 { config, .. } => {
                (config.tie_word_embeddings, config.num_hidden_layers, true)
            }
            Self::Llama { config, .. } => {
                (config.tie_word_embeddings, config.num_hidden_layers, false)
            }
        };
        let mut required = vec![
            "token_embd.weight".to_owned(),
            "output_norm.weight".to_owned(),
        ];
        if !tie_word_embeddings {
            required.push("output.weight".to_owned());
        }
        for layer in 0..layers {
            required.extend([
                format!("blk.{layer}.attn_norm.weight"),
                format!("blk.{layer}.attn_q.weight"),
                format!("blk.{layer}.attn_k.weight"),
                format!("blk.{layer}.attn_v.weight"),
                format!("blk.{layer}.attn_output.weight"),
                format!("blk.{layer}.ffn_norm.weight"),
                format!("blk.{layer}.ffn_gate.weight"),
                format!("blk.{layer}.ffn_up.weight"),
                format!("blk.{layer}.ffn_down.weight"),
            ]);
            if qk_norm {
                required.extend([
                    format!("blk.{layer}.attn_q_norm.weight"),
                    format!("blk.{layer}.attn_k_norm.weight"),
                ]);
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
                "converted checkpoint is missing required {} tensors: {}",
                self.architecture(),
                missing.join(", ")
            ))
        }
    }
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
        "Usage: quantinfer-convert --config config.json --weights model.safetensors --output model.gguf [--architecture qwen3|llama] [--tokenizer tokenizer.json] [--manifest manifest.json] [--quantize q8_0|q4_0|q4_k|q5_k|q6_k] [--ggml]"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{load_tokenizer_metadata, ConversionModel};

    #[test]
    #[cfg(feature = "ggml")]
    fn selects_the_external_q8_provider_explicitly() {
        use super::{q8_provider, QuantizedData, Tensor};

        let zeros = Tensor::from_f32(vec![32], vec![0.0; 32]).unwrap();
        for (use_ggml, expected_scale) in [(false, 1.0), (true, 0.0)] {
            let data = q8_provider(use_ggml).unwrap().quantize(&zeros).unwrap();
            let QuantizedData::Q8_0(quantized) = data else {
                panic!("expected Q8_0 blocks");
            };
            assert_eq!(quantized.blocks()[0].scale, expected_scale);
        }
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

    #[test]
    fn detects_llama_config_and_validates_canonical_tensor_set() {
        let model = ConversionModel::from_config(
            r#"{
                "model_type": "llama",
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "tie_word_embeddings": true,
                "architectures": ["LlamaForCausalLM"]
            }"#,
            None,
        )
        .unwrap();
        assert_eq!(model.architecture(), "llama");
        assert_eq!(model.vocab_size(), 32);
        assert_eq!(
            model.canonical_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".to_owned())
        );
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        model.validate_required_tensors(&names).unwrap();
    }
}
