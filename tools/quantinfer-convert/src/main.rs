use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use quantinfer_core::{
    QuantizationAxis, QuantizationScheme, Shape, Tensor, GGML_OUTPUT_ROWS_LAYOUT,
};
use quantinfer_format::{
    qif::import_gguf, sha256_hex, sha256_reader, ExecutionVariant, GgufReader, GgufSpool,
    GgufValue, GgufWriter, ModelManifest, SafeTensorFileReader, SafeTensorShards, TensorManifest,
};
use quantinfer_model::llama::ggml_qk_channel;
use quantinfer_model::{LlamaAdapter, LlamaConfig, ModelAdapter, Qwen3Adapter, Qwen3Config};
use quantinfer_quant::{
    FloatProvider, ImatrixQ4_0Provider, MixedPrecisionPlan, NativeQ4KProvider, NativeQ4_0Provider,
    NativeQ5KProvider, NativeQ6KProvider, NativeQ8_0Provider, QuantizationProvider, QuantizedData,
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
    if arguments
        .first()
        .is_some_and(|argument| argument == "from-gguf")
    {
        return convert_qif(&arguments[1..]);
    }
    let config_path = required_path(&arguments, "--config")?;
    let weights_path = required_path(&arguments, "--weights")?;
    let weights_checksums = arguments
        .iter()
        .any(|argument| argument == "--weights-checksums")
        .then(|| required_path(&arguments, "--weights-checksums"))
        .transpose()?;
    let output_path = required_path(&arguments, "--output")?;
    let manifest_path = optional_value(&arguments, "--manifest").map(PathBuf::from);
    let tokenizer_path = optional_value(&arguments, "--tokenizer").map(PathBuf::from);
    let requested_architecture = optional_value(&arguments, "--architecture");
    let quantization = optional_value(&arguments, "--quantize")
        .as_deref()
        .map(parse_quantization)
        .transpose()?
        .unwrap_or(QuantizationScheme::None);
    let mixed_precision = arguments
        .iter()
        .any(|argument| argument == "--mixed-precision")
        .then(|| load_mixed_precision(&required_path(&arguments, "--mixed-precision")?))
        .transpose()?;
    let ggml_layout = match optional_value(&arguments, "--gguf-layout").as_deref() {
        None if arguments.iter().any(|argument| argument == "--gguf-layout") => {
            return Err("--gguf-layout requires quantinfer or ggml".to_owned());
        }
        None | Some("quantinfer") => false,
        Some("ggml") => true,
        Some(value) => {
            return Err(format!(
                "unknown GGUF layout {value}; use quantinfer or ggml"
            ))
        }
    };
    let tokenizer_pre = optional_value(&arguments, "--tokenizer-pre");
    let use_ggml = arguments.iter().any(|argument| argument == "--ggml");
    if use_ggml && quantization != QuantizationScheme::Q8_0 {
        return Err("--ggml requires --quantize q8_0".to_owned());
    }
    if mixed_precision.is_some() && (ggml_layout || use_ggml) {
        return Err(
            "--mixed-precision requires the QuantInfer layout and native quantizers".to_owned(),
        );
    }
    let imatrix_path = arguments
        .iter()
        .any(|argument| argument == "--imatrix")
        .then(|| required_path(&arguments, "--imatrix"))
        .transpose()?;
    if imatrix_path.is_some() && (!ggml_layout || quantization != QuantizationScheme::Q4_0) {
        return Err("--imatrix requires --gguf-layout ggml and --quantize q4_0".to_owned());
    }
    let q8_provider = q8_provider(use_ggml)?;

    let config_json = fs::read_to_string(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    let model = ConversionModel::from_config(&config_json, requested_architecture.as_deref())?;
    let ggml_file_type = if ggml_layout {
        if model.architecture() != "llama"
            || tokenizer_path.is_none()
            || tokenizer_pre.as_deref().is_none_or(str::is_empty)
        {
            return Err(
                "GGML layout requires Llama weights, --tokenizer, and --tokenizer-pre".to_owned(),
            );
        }
        let config: Value = serde_json::from_str(&config_json)
            .map_err(|error| format!("cannot parse GGML source configuration: {error}"))?;
        if !config["rope_scaling"].is_null() || config["rope_interleaved"].as_bool() == Some(true) {
            return Err(
                "GGML export requires unscaled, half-split RoPE in the source checkpoint"
                    .to_owned(),
            );
        }
        Some(match quantization {
            QuantizationScheme::None => 0,
            QuantizationScheme::Q4_0 => 2,
            QuantizationScheme::Q8_0 => 7,
            _ => return Err("GGML layout supports F32, Q4_0 and Q8_0 export".to_owned()),
        })
    } else {
        if tokenizer_pre.is_some() {
            return Err("--tokenizer-pre requires --gguf-layout ggml".to_owned());
        }
        None
    };
    let weights = WeightSource::open(&weights_path, weights_checksums.as_deref())?;

    let mut writer = GgufWriter::new();
    model.write_metadata(&mut writer, ggml_file_type)?;
    let mut imatrix = if let Some(path) = imatrix_path {
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        writer.insert_metadata(
            "quantinfer.quantization.imatrix_sha256",
            GgufValue::String(sha256_hex(&bytes)),
        );
        Some(
            load_imatrix(bytes)
                .map_err(|error| format!("cannot parse imatrix {}: {error}", path.display()))?,
        )
    } else {
        None
    };
    if mixed_precision.is_some() {
        writer.insert_metadata(
            "quantinfer.quantization.variant",
            GgufValue::String(QuantizationScheme::Mixed.name().to_owned()),
        );
    }
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
        if let Some(pre) = tokenizer_pre {
            writer.insert_metadata("tokenizer.ggml.pre", GgufValue::String(pre));
            let mut types = tokenizer.token_types;
            types.resize(model.vocab_size(), 5);
            writer.insert_metadata(
                "tokenizer.ggml.token_type",
                GgufValue::Array(types.into_iter().map(GgufValue::I32).collect()),
            );
            writer.insert_metadata(
                "tokenizer.ggml.add_bos_token",
                GgufValue::Bool(tokenizer.add_bos_token),
            );
            writer.insert_metadata(
                "tokenizer.ggml.add_eos_token",
                GgufValue::Bool(tokenizer.add_eos_token),
            );
        }
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
    let output_dir = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let scratch = tempfile::tempfile_in(output_dir).map_err(|error| {
        format!(
            "cannot create GGUF scratch file in {}: {error}",
            output_dir.display()
        )
    })?;
    let mut spool = GgufSpool::new(scratch, writer);
    let mut written_names = BTreeSet::new();
    let mut manifest_tensors = Vec::new();
    let mut projection_schemes = Vec::new();
    for (name, shape) in weights.tensors()? {
        let values = weights
            .tensor_f32(&name)
            .map_err(|error| format!("cannot decode tensor {name}: {error}"))?;
        let canonical_name = model.canonical_tensor_name(&name);
        if ggml_layout && canonical_name.is_none() {
            return Err(format!("unsupported source tensor {name} for GGML export"));
        }
        let output_name = canonical_name.as_deref().unwrap_or(&name);
        if !written_names.insert(output_name.to_owned()) {
            return Err(format!(
                "multiple SafeTensors entries map to GGUF tensor {output_name}"
            ));
        }
        let (shape, values) = if ggml_layout {
            model.ggml_tensor_layout(output_name, &shape, values)?
        } else {
            model.canonicalize_tensor_layout(output_name, &shape, values)?
        };
        let quantizable_width = if ggml_layout {
            shape.first().copied().unwrap_or(0)
        } else {
            values.len()
        };
        let assignment = mixed_precision
            .as_ref()
            .and_then(|plan| plan.scheme_for(output_name));
        let requested_scheme = assignment.unwrap_or(quantization);
        let block_size = match requested_scheme {
            QuantizationScheme::None => 1,
            QuantizationScheme::Q8_0 => quantinfer_core::Q8_0_BLOCK_SIZE,
            QuantizationScheme::Q4_0 => quantinfer_core::Q4_0_BLOCK_SIZE,
            QuantizationScheme::Q4_K | QuantizationScheme::Q5_K | QuantizationScheme::Q6_K => {
                quantinfer_core::Q4_K_BLOCK_SIZE
            }
            QuantizationScheme::Mixed => {
                unreachable!("mixed precision must select a tensor scheme")
            }
        };
        let is_projection = is_quantizable_linear(output_name);
        let can_use_scheme = is_projection && quantizable_width.is_multiple_of(block_size);
        if assignment.is_some() && !can_use_scheme {
            return Err(format!(
                "mixed precision assignment for {output_name} requires a projection aligned to {block_size} elements"
            ));
        }
        let scheme = if can_use_scheme {
            requested_scheme
        } else {
            QuantizationScheme::None
        };
        if mixed_precision.is_some() && is_projection && !projection_schemes.contains(&scheme) {
            projection_schemes.push(scheme);
        }
        let importance =
            if let (Some(weights), QuantizationScheme::Q4_0) = (&mut imatrix, scheme) {
                Some(weights.remove(output_name).ok_or_else(|| {
                    format!("imatrix has no input channel weights for {output_name}")
                })?)
            } else {
                None
            };
        let calibrated = importance
            .as_deref()
            .map(ImatrixQ4_0Provider::new)
            .transpose()
            .map_err(|error| format!("invalid imatrix weights for {output_name}: {error}"))?;
        let provider: &dyn QuantizationProvider = match scheme {
            QuantizationScheme::None => &FloatProvider,
            QuantizationScheme::Q8_0 => q8_provider,
            QuantizationScheme::Q4_0 => match &calibrated {
                Some(provider) => provider,
                None => &NativeQ4_0Provider,
            },
            QuantizationScheme::Q4_K => &NativeQ4KProvider,
            QuantizationScheme::Q5_K => &NativeQ5KProvider,
            QuantizationScheme::Q6_K => &NativeQ6KProvider,
            QuantizationScheme::Mixed => {
                unreachable!("mixed precision must select a tensor scheme")
            }
        };
        let mut descriptor = provider
            .storage_descriptor(Shape::from(shape.clone()))
            .map_err(|error| format!("cannot describe tensor {name}: {error}"))?;
        if ggml_layout && shape.len() == 2 {
            if output_name == "token_embd.weight" {
                descriptor.logical_shape = Shape::from(vec![shape[1], shape[0]]);
            } else {
                descriptor.physical_layout = GGML_OUTPUT_ROWS_LAYOUT.to_owned();
                if scheme != QuantizationScheme::None {
                    descriptor.quantization_axis = QuantizationAxis::Axis(0);
                }
            }
        }
        manifest_tensors.push(TensorManifest {
            name: output_name.to_owned(),
            descriptor,
            checksum_sha256: None,
        });
        let write_result = if scheme == QuantizationScheme::None {
            spool.append(|writer| writer.add_f32_tensor(output_name, shape, &values))
        } else {
            let tensor = Tensor::from_f32(shape, values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = provider
                .quantize(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            if quantized.scheme() != scheme {
                return Err(format!(
                    "{} provider returned a different scheme for {name}",
                    scheme.name()
                ));
            }
            spool.append(|writer| {
                match &quantized {
                    QuantizedData::Q8_0(data) => writer.add_q8_0_tensor(output_name, data),
                    QuantizedData::Q4_0(data) => writer.add_q4_0_tensor(output_name, data),
                    QuantizedData::Q4_K(data) => writer.add_q4_k_tensor(output_name, data),
                    QuantizedData::Q5_K(data) => writer.add_q5_k_tensor(output_name, data),
                    QuantizedData::Q6_K(data) => writer.add_q6_k_tensor(output_name, data),
                    QuantizedData::Float(_) => unreachable!("provider scheme was checked"),
                }
                Ok(())
            })
        };
        write_result.map_err(|error| format!("cannot add tensor {name}: {error}"))?;
    }
    model.validate_required_tensors(&written_names)?;
    if let Some(name) = imatrix.as_ref().and_then(|weights| weights.keys().next()) {
        return Err(format!("imatrix contains an unused tensor {name}"));
    }
    if let Some(plan) = &mixed_precision {
        if let Some(unknown) = plan
            .assignments()
            .keys()
            .find(|name| !written_names.contains(*name))
        {
            return Err(format!(
                "mixed precision assignment names unknown tensor {unknown}"
            ));
        }
        if projection_schemes.len() < 2 {
            return Err(
                "mixed precision requires at least two schemes across the projection weights"
                    .to_owned(),
            );
        }
    }
    #[cfg(unix)]
    let output = tempfile::Builder::new()
        .permissions(fs::Permissions::from_mode(0o666))
        .tempfile_in(output_dir);
    #[cfg(not(unix))]
    let output = tempfile::NamedTempFile::new_in(output_dir);
    let mut output =
        output.map_err(|error| format!("cannot prepare {}: {error}", output_path.display()))?;
    spool
        .finish(output.as_file_mut())
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    let checksum = if manifest_path.is_some() {
        let file = output.as_file_mut();
        file.rewind()
            .and_then(|()| sha256_reader(file))
            .map(Some)
            .map_err(|error| format!("cannot hash {}: {error}", output_path.display()))?
    } else {
        None
    };
    if let Ok(metadata) = fs::metadata(&output_path) {
        output
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| {
                format!(
                    "cannot set permissions on {}: {error}",
                    output_path.display()
                )
            })?;
    }
    output
        .persist(&output_path)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    if let (Some(manifest_path), Some(checksum)) = (manifest_path, checksum) {
        let mut manifest = ModelManifest::new(model.architecture(), model.architecture());
        manifest.source_format = "safetensors-to-gguf".to_owned();
        manifest.quantization = if mixed_precision.is_some() {
            QuantizationScheme::Mixed
        } else {
            quantization
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
        manifest.checksums.insert(output_name.to_owned(), checksum);
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

fn parse_quantization(value: &str) -> Result<QuantizationScheme, String> {
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

fn load_mixed_precision(path: &Path) -> Result<MixedPrecisionPlan, String> {
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

enum WeightSource {
    Single(SafeTensorFileReader),
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

    fn ggml_tensor_layout(
        &self,
        name: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let Self::Llama { config, .. } = self else {
            return Err("GGML layout requires Llama weights".to_owned());
        };
        match shape {
            [_] => Ok((shape.to_vec(), values)),
            &[output, input] if output != 0 && input != 0 => {
                let expected_outputs = if name.ends_with(".attn_q.weight") {
                    Some(config.query_projection_size())
                } else if name.ends_with(".attn_k.weight") {
                    Some(config.key_value_projection_size())
                } else {
                    None
                };
                let shape = vec![input, output];
                let Some(expected) = expected_outputs else {
                    return Ok((shape, values));
                };
                if output != expected || input != config.hidden_size {
                    return Err(format!("invalid shape for GGML RoPE projection {name}"));
                }
                let head_dim = config.effective_head_dim();
                let mut ordered = Vec::with_capacity(values.len());
                for row in 0..output {
                    let start = ggml_qk_channel(row, head_dim) * input;
                    ordered.extend_from_slice(&values[start..start + input]);
                }
                Ok((shape, ordered))
            }
            _ => Err(format!("invalid tensor shape for GGML export at {name}")),
        }
    }

    fn write_metadata(
        &self,
        writer: &mut GgufWriter,
        ggml_file_type: Option<u32>,
    ) -> Result<(), String> {
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
        if let Some(file_type) = ggml_file_type {
            writer.insert_metadata("general.type", GgufValue::String("model".to_owned()));
            writer.insert_metadata("general.file_type", GgufValue::U32(file_type));
            writer.insert_metadata("general.quantization_version", GgufValue::U32(2));
            for key in [
                "attention.key_length",
                "attention.value_length",
                "rope.dimension_count",
            ] {
                writer.insert_metadata(
                    format!("{prefix}.{key}"),
                    GgufValue::U32(metadata_u32(head_dim, key)?),
                );
            }
        } else {
            writer.insert_metadata(
                format!("{prefix}.tie_word_embeddings"),
                GgufValue::Bool(tie_word_embeddings),
            );
        }
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
    fn open(path: &Path, checksums_path: Option<&Path>) -> Result<Self, String> {
        let checksums = checksums_path.map(load_checksums).transpose()?;
        if let Some(checksums) = &checksums {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("invalid SafeTensors path {}", path.display()))?;
            let expected = checksums
                .get(name)
                .ok_or_else(|| format!("missing checksum for {name}"))?;
            let actual = fs::File::open(path)
                .and_then(|mut file| sha256_reader(&mut file))
                .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
            if &actual != expected {
                return Err(format!(
                    "checksum mismatch for {}: expected {expected}, got {actual}",
                    path.display()
                ));
            }
        }
        if path.to_string_lossy().ends_with(".index.json") {
            let shards = SafeTensorShards::open(path)
                .map_err(|error| format!("cannot parse SafeTensors index: {error}"))?;
            if let Some(checksums) = &checksums {
                shards
                    .verify_checksums(checksums)
                    .map_err(|error| error.to_string())?;
            }
            Ok(Self::Sharded(shards))
        } else {
            SafeTensorFileReader::open(path)
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

fn load_checksums(path: &Path) -> Result<BTreeMap<String, String>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut checksums = BTreeMap::new();
    for (index, line) in content.lines().enumerate() {
        let invalid = || {
            format!(
                "invalid SHA-256 entry in {} at line {}",
                path.display(),
                index + 1
            )
        };
        let (hash, filename) = line.split_once(' ').ok_or_else(invalid)?;
        let filename = filename.strip_prefix([' ', '*']).ok_or_else(invalid)?;
        let filename = filename.strip_prefix("./").unwrap_or(filename);
        if hash.len() != 64
            || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            || filename.is_empty()
        {
            return Err(invalid());
        }
        if checksums
            .insert(filename.to_owned(), hash.to_ascii_lowercase())
            .is_some()
        {
            return Err(format!(
                "duplicate checksum for {filename} in {}",
                path.display()
            ));
        }
    }
    Ok(checksums)
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
    token_types: Vec<i32>,
    add_bos_token: bool,
    add_eos_token: bool,
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
    let mut token_types = vec![1; token_count];
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
                token_types.resize(token_id + 1, 1);
            }
            tokens[token_id] = Some(token.to_owned());
            token_types[token_id] =
                if added_token.get("special").and_then(Value::as_bool) == Some(true) {
                    3 // GGML control token
                } else {
                    4 // GGML user-defined token
                };
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
        .map_err(|error| format!("cannot parse model config for tokenizer ids: {error}"))?;
    let bos_token_id = token_id_from_value(config.get("bos_token_id"));
    let eos_token_id = token_id_from_value(config.get("eos_token_id"));
    let template = root["post_processor"]["single"].as_array();
    let matches_special = |entry: Option<&Value>, token: Option<usize>| {
        let Some(text) = token.and_then(|id| tokens.get(id)) else {
            return false;
        };
        entry.and_then(|entry| entry["SpecialToken"]["id"].as_str()) == Some(text.as_str())
    };
    let add_bos_token = matches_special(template.and_then(|items| items.first()), bos_token_id);
    let add_eos_token = matches_special(template.and_then(|items| items.last()), eos_token_id);
    Ok(TokenizerMetadata {
        tokens,
        merges,
        bos_token_id,
        eos_token_id,
        token_types,
        add_bos_token,
        add_eos_token,
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

fn load_imatrix(bytes: Vec<u8>) -> Result<BTreeMap<String, Vec<f32>>, String> {
    if !bytes.starts_with(b"GGUF") {
        return serde_json::from_slice(&bytes).map_err(|error| error.to_string());
    }
    let reader = GgufReader::open(bytes).map_err(|error| error.to_string())?;
    if !matches!(reader.metadata().get("general.type"), Some(GgufValue::String(kind)) if kind == "imatrix")
    {
        return Err("GGUF artifact does not declare imatrix data".to_owned());
    }
    let mut weights = BTreeMap::new();
    for tensor in reader.tensors() {
        let Some(name) = tensor.name.strip_suffix(".in_sum2") else {
            continue;
        };
        if !name.starts_with("blk.") || !is_quantizable_linear(name) {
            continue;
        }
        if !matches!(tensor.shape.as_slice(), [_] | [_, 1]) {
            return Err(format!("imatrix {name} must contain one dense input row"));
        }
        let counts = reader
            .tensor_f32(&format!("{name}.counts"))
            .map_err(|error| error.to_string())?;
        let count = match counts.as_f32().map_err(|error| error.to_string())? {
            [count] if count.is_finite() && *count > 0.0 => *count,
            _ => {
                return Err(format!(
                    "imatrix {name} must contain one positive observation count"
                ))
            }
        };
        let sums = reader
            .tensor_f32(&tensor.name)
            .map_err(|error| error.to_string())?;
        let means = sums
            .as_f32()
            .map_err(|error| error.to_string())?
            .iter()
            .map(|sum| sum / count)
            .collect();
        if weights.insert(name.to_owned(), means).is_some() {
            return Err(format!("imatrix contains duplicate projections for {name}"));
        }
    }
    Ok(weights)
}

fn convert_qif(arguments: &[String]) -> Result<(), String> {
    let input = required_path(arguments, "--input")?;
    let output = required_path(arguments, "--output")?;
    let model_id = optional_value(arguments, "--model-id").unwrap_or_else(|| {
        input
            .file_stem()
            .unwrap_or(input.as_os_str())
            .to_string_lossy()
            .into_owned()
    });
    let bytes =
        fs::read(&input).map_err(|error| format!("cannot read {}: {error}", input.display()))?;
    let reader = GgufReader::open(bytes).map_err(|error| error.to_string())?;
    let directory = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(directory)
        .map_err(|error| format!("cannot prepare {}: {error}", output.display()))?;
    import_gguf(&reader, model_id, temporary.as_file_mut()).map_err(|error| error.to_string())?;
    temporary
        .persist_noclobber(&output)
        .map_err(|error| format!("cannot publish {}: {error}", output.display()))?;
    Ok(())
}

fn print_help() {
    println!(
        "Usage: quantinfer-convert --config config.json --weights model.safetensors|model.safetensors.index.json --output model.gguf [--weights-checksums checksums.sha256] [--architecture qwen3|llama] [--tokenizer tokenizer.json] [--manifest manifest.json] [--quantize f32|q8_0|q4_0|q4_k|q5_k|q6_k] [--mixed-precision PLAN.json] [--imatrix CHANNEL_WEIGHTS.json|IMATRIX.gguf] [--ggml] [--gguf-layout quantinfer|ggml] [--tokenizer-pre NAME]\n       quantinfer-convert from-gguf --input model.gguf --output model.qif [--model-id NAME]"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{load_imatrix, load_tokenizer_metadata, ConversionModel, GgufValue, GgufWriter};

    #[test]
    fn reads_dense_llama_imatrix_and_normalizes_observations() {
        let mut writer = GgufWriter::new();
        writer.insert_metadata("general.type", GgufValue::String("imatrix".to_owned()));
        writer
            .add_f32_tensor("blk.0.attn_q.weight.in_sum2", vec![2, 1], &[4.0, 12.0])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_q.weight.counts", vec![1], &[4.0])
            .unwrap();
        let weights = load_imatrix(writer.finish().unwrap()).unwrap();
        assert_eq!(weights["blk.0.attn_q.weight"], [1.0, 3.0]);
        assert!(load_imatrix(GgufWriter::new().finish().unwrap()).is_err());
    }

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
