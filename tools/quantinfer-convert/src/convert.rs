//! Conversion command orchestration.

use std::collections::BTreeSet;
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
    GgufValue, GgufWriter, ModelManifest, TensorManifest,
};
use quantinfer_quant::{
    FloatProvider, ImatrixQ4_0Provider, NativeQ4KProvider, NativeQ4_0Provider, NativeQ5KProvider,
    NativeQ6KProvider, QuantizationProvider, QuantizedData,
};
use serde_json::Value;

use super::common::{
    is_quantizable_linear, load_imatrix, metadata_u32, optional_value, required_path,
};
use super::model::ConversionModel;
use super::policy::{
    load_mixed_precision, load_mixed_precision_policy, parse_quantization, q8_provider,
};
use super::source::WeightSource;
use super::tokenizer::load_tokenizer_metadata;
use crate::print_help;

pub(crate) fn run() -> Result<(), String> {
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
    let mixed_precision_policy = arguments
        .iter()
        .any(|argument| argument == "--mixed-precision-policy")
        .then(|| {
            load_mixed_precision_policy(&required_path(&arguments, "--mixed-precision-policy")?)
        })
        .transpose()?;
    if mixed_precision.is_some() && mixed_precision_policy.is_some() {
        return Err(
            "--mixed-precision and --mixed-precision-policy are mutually exclusive".to_owned(),
        );
    }
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
    if (mixed_precision.is_some() || mixed_precision_policy.is_some()) && (ggml_layout || use_ggml)
    {
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
        if tokenizer_path.is_none() || tokenizer_pre.as_deref().is_none_or(str::is_empty) {
            return Err("GGML layout requires --tokenizer and --tokenizer-pre".to_owned());
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
    if mixed_precision.is_some() || mixed_precision_policy.is_some() {
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
        let is_projection = is_quantizable_linear(output_name);
        let assignment = mixed_precision
            .as_ref()
            .and_then(|plan| plan.scheme_for(output_name))
            .or_else(|| {
                mixed_precision_policy
                    .as_ref()
                    .and_then(|policy| policy.scheme_for(output_name))
            });
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
        if (mixed_precision.is_some() || mixed_precision_policy.is_some())
            && is_projection
            && !projection_schemes.contains(&scheme)
        {
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

pub(crate) fn convert_qif(arguments: &[String]) -> Result<(), String> {
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
