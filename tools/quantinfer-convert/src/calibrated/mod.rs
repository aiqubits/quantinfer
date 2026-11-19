//! Sequential, activation-calibrated whole-model quantization workflow.

#[cfg(feature = "awq")]
pub(crate) mod awq;
mod dataset;
#[cfg(feature = "gptq")]
mod gptq;
#[cfg(feature = "gptq")]
pub(crate) use gptq::run;
mod model;
mod options;
mod packing;
mod source;
mod validation;

use options::WorkflowOptions;
use quantinfer::{
    cpu::{load_gguf_tokenizer, CpuModel, ModelSource},
    tokenizer::Tokenizer,
};
use quantinfer_format::{qif, ExecutionVariant};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

pub(super) struct ProductionContext<'a> {
    pub model: &'a mut CpuModel,
    pub manifest: &'a mut quantinfer_format::ModelManifest,
    pub spec: &'a quantinfer::model::ModelSpec,
    pub prompts: &'a [Vec<usize>],
    pub context: usize,
    pub ggml: bool,
    pub statistics_bytes: usize,
}

pub(super) struct Production {
    pub payloads: BTreeMap<String, Vec<u8>>,
    pub records: Vec<Value>,
    pub details: BTreeMap<String, Value>,
}

trait ModelQuantizer {
    fn name(&self) -> &'static str;
    fn id(&self) -> &'static str;
    fn bits(&self) -> usize;
    fn scheme(&self) -> quantinfer_core::QuantizationScheme;
    fn parameters(&self) -> Value;
    fn produce(&self, context: ProductionContext<'_>) -> Result<Production, String>;
}

fn execute(options: WorkflowOptions, algorithm: &impl ModelQuantizer) -> Result<(), String> {
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
    let Production {
        payloads,
        records,
        details,
    } = algorithm.produce(ProductionContext {
        model: &mut model,
        manifest: &mut manifest,
        spec: &spec,
        prompts: &prompts,
        context,
        ggml,
        statistics_bytes: options.statistics_bytes,
    })?;
    let actual = validation::capture(&model, &holdouts, context, spec.vocabulary.eos_token_id)?;
    let quality = validation::compare(
        &baseline,
        &actual,
        options.maximum_kl,
        options.minimum_top10,
    )?;
    manifest.model_id = options.model_id.clone().unwrap_or_else(|| {
        format!(
            "{}-{}{}",
            options
                .input
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            algorithm.name(),
            algorithm.bits()
        )
    });
    manifest.quantization = algorithm.scheme();
    manifest.execution_variants = vec![ExecutionVariant {
        name: format!("{}-cpu", algorithm.name()),
        quantization: manifest.quantization,
        dtype: "f32-accumulation".into(),
        runtime_provider: "cpu-scalar".into(),
        kernel_provider: "cpu-scalar".into(),
    }];
    metadata.insert(
        "quantinfer.quantization.algorithm".into(),
        qif::MetadataValue::String(algorithm.id().into()),
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
    let mut report = json!({"schema_version":1,"algorithm":algorithm.id(),"input":options.input,"output":options.output,
        "source":identity,"calibration_dataset_sha256":dataset.sha256,"calibration_prompts":prompts.len(),"calibration_token_rows":prompts.iter().map(Vec::len).sum::<usize>(),
        "layers":spec.layers.len(),"quantized_projections":records.len(),"projections":records,
        "validation_dataset_sha256":validation_dataset.as_ref().map(|d|d.sha256.as_str()),"quality_status":if holdouts.is_empty(){"not_evaluated"}else{"passed"},
        "quality_thresholds":{"maximum_kl":options.maximum_kl,"minimum_top10_overlap":options.minimum_top10},"holdouts":quality,"artifact_parity":true,"payload_verified":true});
    if let Value::Object(parameters) = algorithm.parameters() {
        report
            .as_object_mut()
            .expect("report object")
            .extend(parameters);
    }
    report
        .as_object_mut()
        .expect("report object")
        .extend(details);
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
    println!(
        "{}_quantized_projections={}",
        algorithm.name(),
        records.len()
    );
    println!(
        "{}_quality_status={}",
        algorithm.name(),
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
