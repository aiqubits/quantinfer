//! Sequential, activation-calibrated whole-model quantization workflow.

mod dataset;
mod model;
mod options;
mod packing;
mod source;
mod validation;

use options::WorkflowOptions;
use quantinfer::{
    cpu::{load_gguf_tokenizer, CpuModel, ModelSource},
    model::TensorRole,
    runtime::{dense_storage_name, LinearWeight},
    tokenizer::Tokenizer,
};
use quantinfer_adapter_gptq::HessianCalibration;
use quantinfer_format::{qif, ExecutionVariant};
use quantinfer_quant::{CalibratedQuantizationProvider, QuantizationProvider};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    let options = WorkflowOptions::parse(arguments)?;
    let dataset = dataset::read_dataset(&options.dataset, options.dataset_bytes)?;
    let source = ModelSource::open(&options.input).map_err(|e| e.to_string())?;
    let reader = source.open_reader(None).map_err(|e| e.to_string())?;
    let package_manifest = source.package().map(|p| &p.manifest);
    let (mut manifest, mut metadata, identity) = source::describe(&reader, package_manifest)?;
    let spec = CpuModel::spec_from_gguf(&reader, package_manifest).map_err(|e| e.to_string())?;
    let tokenizer = load_gguf_tokenizer(&reader).map_err(|e| e.to_string())?;
    let mut model =
        CpuModel::from_gguf(&reader, package_manifest, None).map_err(|e| e.to_string())?;
    let context = options.context.unwrap_or(spec.max_sequence_length.min(512));
    if context == 0 || context > spec.max_sequence_length {
        return Err("calibration context is outside the model capacity".into());
    }
    let tokenize = |texts: &[String]| -> Result<Vec<Vec<usize>>, String> {
        texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let tokens = tokenizer.encode(text).map_err(|e| e.to_string())?;
                if tokens.is_empty() || tokens.len() > context {
                    return Err(format!(
                        "dataset sample {} needs 1 to {context} tokens",
                        index + 1
                    ));
                }
                Ok(tokens)
            })
            .collect()
    };
    let prompts = tokenize(&dataset.texts)?;
    let validation_dataset = options
        .validation
        .as_ref()
        .map(|p| dataset::read_dataset(p, options.dataset_bytes))
        .transpose()?;
    let holdouts = validation_dataset
        .as_ref()
        .map(|d| tokenize(&d.texts))
        .transpose()?
        .unwrap_or_default();
    if holdouts.len() > options.maximum_validation_prompts {
        return Err("holdout dataset exceeds --max-validation-prompts".into());
    }
    if !holdouts.is_empty() && options.minimum_top10 > spec.vocabulary.size.min(10) {
        return Err("--minimum-top10-overlap exceeds the available vocabulary".into());
    }
    let baseline = validation::capture(&model, &holdouts, context, spec.vocabulary.eos_token_id)?;
    let ggml = reader.uses_ggml_output_rows().map_err(|e| e.to_string())?;
    let mut payloads = BTreeMap::<String, Vec<u8>>::new();
    let mut records = Vec::new();
    for layer in 0..spec.layers.len() {
        for roles in model::GROUPS {
            let input = model::projection(&model, layer, roles[0])?.shape().dims()[0];
            let mut calibration = HessianCalibration::new(input, options.statistics_bytes)
                .map_err(|e| e.to_string())?;
            for tokens in &prompts {
                let mut observer =
                    |observed_layer,
                     observed_roles: &[TensorRole],
                     activations: &quantinfer_core::Tensor| {
                        if observed_layer == layer
                            && observed_roles.iter().any(|role| roles.contains(role))
                        {
                            calibration.observe_rows(activations).map_err(|e| {
                                quantinfer::runtime::RuntimeError::InvalidInput(e.to_string())
                            })?;
                        }
                        Ok(())
                    };
                model
                    .observe_projections(
                        tokens,
                        &mut model.new_cache(context).map_err(|e| e.to_string())?,
                        &mut observer,
                    )
                    .map_err(|e| e.to_string())?;
            }
            for &role in roles {
                let name = dense_storage_name(role, Some(layer));
                let LinearWeight::F32(weights) = model::projection(&model, layer, role)? else {
                    return Err(format!("{name} is not a floating projection"));
                };
                let quantized = options
                    .provider
                    .quantize_calibrated(weights, &calibration)
                    .map_err(|e| format!("cannot quantize {name}: {e}"))?;
                let error = options
                    .provider
                    .error_report(weights, &quantized)
                    .map_err(|e| e.to_string())?;
                let rope_dim = (ggml
                    && spec.architecture == "llama"
                    && matches!(
                        role,
                        TensorRole::QueryProjection | TensorRole::KeyProjection
                    ))
                .then_some(spec.layers[layer].attention.head_dim);
                let raw = packing::encoded_projection(&quantized, &name, rope_dim)?;
                let tensor = manifest
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == name)
                    .ok_or_else(|| format!("manifest is missing {name}"))?;
                let mut descriptor = options
                    .provider
                    .storage_descriptor(weights.shape().clone())
                    .map_err(|e| e.to_string())?;
                descriptor.logical_shape = tensor.descriptor.logical_shape.clone();
                tensor.descriptor = descriptor;
                tensor.checksum_sha256 = None;
                records.push(json!({"tensor":name,"layer":layer,"role":format!("{role:?}"),"calibration_rows":calibration.rows(),
                    "source_core_f32_sha256":validation::tensor_hash(weights)?,"payload_sha256":quantinfer_format::sha256_hex(&raw),
                    "maximum_weight_error":error.max_absolute_error,"mean_weight_error":error.mean_absolute_error,"relative_l2":error.relative_l2_error}));
                payloads.insert(name.clone(), raw);
                model::replace_projection(
                    &mut model,
                    layer,
                    role,
                    packing::runtime_weight(quantized)?,
                )?;
                println!("gptq_projection={name}");
            }
        }
    }
    let actual = validation::capture(&model, &holdouts, context, spec.vocabulary.eos_token_id)?;
    let quality = validation::compare(
        &baseline,
        &actual,
        options.maximum_kl,
        options.minimum_top10,
    )?;
    manifest.model_id = options.model_id.clone().unwrap_or_else(|| {
        format!(
            "{}-gptq{}",
            options
                .input
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            options.provider.options().bits
        )
    });
    manifest.quantization = options.provider.scheme();
    manifest.execution_variants = vec![ExecutionVariant {
        name: "gptq-cpu".into(),
        quantization: manifest.quantization,
        dtype: "f32-accumulation".into(),
        runtime_provider: "cpu-scalar".into(),
        kernel_provider: "cpu-scalar".into(),
    }];
    metadata.insert(
        "quantinfer.quantization.algorithm".into(),
        qif::MetadataValue::String("gptq-static-signed-f16".into()),
    );
    metadata.insert(
        "quantinfer.quantization.variant".into(),
        qif::MetadataValue::String(manifest.quantization.name().into()),
    );
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staging = tempfile::tempdir_in(parent).map_err(|e| e.to_string())?;
    let prepared = staging.path().join("artifact");
    options.qif.publish_tensors(
        manifest,
        &metadata,
        |name| {
            if let Some(bytes) = payloads.get(name) {
                Ok(bytes.as_slice())
            } else {
                reader.tensor_bytes(name)
            }
        },
        &prepared,
    )?;
    let artifact = ModelSource::open(&prepared).map_err(|e| e.to_string())?;
    let artifact_reader = artifact.open_reader(None).map_err(|e| e.to_string())?;
    artifact_reader.verify_qif().map_err(|e| e.to_string())?;
    for (name, bytes) in &payloads {
        if artifact_reader
            .tensor_data(name)
            .map_err(|e| e.to_string())?
            .as_ref()
            != bytes.as_slice()
        {
            return Err(format!("published bytes changed for {name}"));
        }
    }
    let reloaded = CpuModel::from_gguf(&artifact_reader, None, None).map_err(|e| e.to_string())?;
    let parity_prompts = if holdouts.is_empty() {
        vec![prompts[0].clone()]
    } else {
        holdouts.clone()
    };
    let parity_reference = if holdouts.is_empty() {
        validation::capture(
            &model,
            &parity_prompts,
            context,
            spec.vocabulary.eos_token_id,
        )?
    } else {
        actual
    };
    let parity = validation::capture(
        &reloaded,
        &parity_prompts,
        context,
        spec.vocabulary.eos_token_id,
    )?;
    validation::verify_parity(&parity_reference, &parity)?;
    drop(reloaded);
    drop(artifact_reader);
    let report = json!({"schema_version":1,"algorithm":"gptq-static-signed-f16","input":options.input,"output":options.output,
        "source":identity,"calibration_dataset_sha256":dataset.sha256,"calibration_prompts":prompts.len(),"calibration_token_rows":prompts.iter().map(Vec::len).sum::<usize>(),
        "bits":options.provider.options().bits,"group_size":options.provider.options().group_size,"damping":options.provider.options().damping,
        "activation_order":options.provider.options().activation_order,"layers":spec.layers.len(),"quantized_projections":records.len(),"projections":records,
        "validation_dataset_sha256":validation_dataset.as_ref().map(|d|d.sha256.as_str()),"quality_status":if holdouts.is_empty(){"not_evaluated"}else{"passed"},
        "quality_thresholds":{"maximum_kl":options.maximum_kl,"minimum_top10_overlap":options.minimum_top10},"holdouts":quality,"artifact_parity":true,"payload_verified":true});
    // The complete candidate is verified while private. Publication is the last
    // artifact operation; an optional report is a separate sidecar.
    if prepared.is_dir() {
        if fs::symlink_metadata(&options.output).is_ok() {
            return Err("output became occupied during quantization".into());
        }
        fs::rename(&prepared, &options.output).map_err(|e| e.to_string())?;
    } else {
        tempfile::TempPath::try_from_path(&prepared)
            .map_err(|e| e.to_string())?
            .persist_noclobber(&options.output)
            .map_err(|e| e.to_string())?;
    }
    if let Some(path) = &options.report {
        write_report(path, &report)?;
    }
    println!("gptq_quantized_projections={}", payloads.len());
    println!(
        "gptq_quality_status={}",
        if holdouts.is_empty() {
            "not_evaluated"
        } else {
            "passed"
        }
    );
    Ok(())
}

fn write_report(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), value).map_err(|e| e.to_string())?;
    temporary.write_all(b"\n").map_err(|e| e.to_string())?;
    temporary
        .persist_noclobber(path)
        .map_err(|e| format!("artifact published, report could not be saved: {e}"))?;
    Ok(())
}
