use std::collections::{btree_map::Entry, BTreeMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use quantinfer::cpu::{load_gguf_tokenizer, CpuModel, ModelSource};
use quantinfer::model::TensorRole;
use quantinfer::quant::ImatrixStatistics;
use quantinfer_core::Tensor;
use quantinfer_runtime::{Result as RuntimeResult, RuntimeError};
use quantinfer_tokenizer::Tokenizer;

use crate::commands::common::option_value;

pub fn collect_imatrix(arguments: &[String]) -> Result<(), String> {
    let path = |flag| {
        option_value(arguments, flag)?
            .map(PathBuf::from)
            .ok_or_else(|| format!("collect-imatrix requires {flag}"))
    };
    let source = path("--input")?;
    let dataset = path("--dataset")?;
    let output = path("--output")?;
    if output.exists() {
        return Err(format!(
            "calibration output already exists at {}",
            output.display()
        ));
    }
    let reader = BufReader::new(
        File::open(&dataset)
            .map_err(|error| format!("cannot open {}: {error}", dataset.display()))?,
    );
    let context = option_value(arguments, "--max-sequence-length")?
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid --max-sequence-length: {error}"))?;

    let source = ModelSource::open(source).map_err(|error| error.to_string())?;
    let weights = source
        .open_reader(None)
        .map_err(|error| error.to_string())?;
    let manifest = source.package().map(|package| &package.manifest);
    let spec = CpuModel::spec_from_gguf(&weights, manifest).map_err(|error| error.to_string())?;
    let tokenizer = load_gguf_tokenizer(&weights).map_err(|error| error.to_string())?;
    let model = CpuModel::from_gguf(&weights, manifest, None).map_err(|error| error.to_string())?;
    drop(weights);
    let maximum = model.max_position_embeddings();
    let context = context.unwrap_or(maximum.min(512));
    if context == 0 || context > maximum {
        return Err(format!(
            "--max-sequence-length must be between 1 and {maximum}"
        ));
    }
    let mut groups = BTreeMap::<Vec<&str>, ImatrixStatistics>::new();
    let mut observer = |layer, roles: &[TensorRole], activations: &Tensor| -> RuntimeResult<()> {
        let names = roles
            .iter()
            .map(|&role| {
                spec.tensor(role, Some(layer))
                    .map(|tensor| tensor.source_name.as_str())
                    .ok_or_else(|| {
                        calibration_error(format!("layer {layer} has no {role:?} tensor"))
                    })
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        let stats = match groups.entry(names) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let channels = activations.shape().last_dim().ok_or_else(|| {
                    calibration_error("projection activations are missing channel dimensions")
                })?;
                entry.insert(ImatrixStatistics::new(channels).map_err(calibration_error)?)
            }
        };
        stats.observe_rows(activations).map_err(calibration_error)
    };

    let mut sample_count = 0;
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.map_err(|error| format!("cannot read calibration line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid calibration JSON at line {line_number}: {error}"))?;
        let text = value
            .as_str()
            .or_else(|| value.get("text").and_then(serde_json::Value::as_str))
            .ok_or_else(|| format!("calibration line {line_number} requires a text string"))?;
        let tokens = tokenizer
            .encode(text)
            .map_err(|error| format!("cannot tokenize calibration line {line_number}: {error}"))?;
        if tokens.is_empty() || tokens.len() > context {
            return Err(format!(
                "calibration line {line_number} has {} tokens and requires 1 to {context}",
                tokens.len()
            ));
        }
        let mut cache = model
            .new_cache(tokens.len())
            .map_err(|error| error.to_string())?;
        model
            .observe_projections(&tokens, &mut cache, &mut observer)
            .map_err(|error| format!("cannot calibrate line {line_number}: {error}"))?;
        sample_count += 1;
    }
    if sample_count == 0 {
        return Err("calibration dataset contains no prompts".to_owned());
    }

    let mut weights = BTreeMap::new();
    for (names, stats) in groups {
        let mean_square = stats.mean_square();
        if mean_square.iter().any(|value| !value.is_finite())
            || !mean_square.iter().any(|&value| value > 0.0)
        {
            return Err(format!(
                "invalid activation weights for {}",
                names.join(", ")
            ));
        }
        for name in names {
            weights.insert(name, mean_square.clone());
        }
    }
    write_calibration(&output, &weights)?;
    println!("calibration_samples={sample_count}");
    println!("calibration_tensors={}", weights.len());
    Ok(())
}

fn write_calibration(path: &Path, weights: &BTreeMap<&str, Vec<f32>>) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut output = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| format!("cannot prepare {}: {error}", path.display()))?;
    serde_json::to_writer(output.as_file_mut(), weights).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())?;
    output
        .persist_noclobber(path)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(())
}

fn calibration_error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::InvalidInput(error.to_string())
}
