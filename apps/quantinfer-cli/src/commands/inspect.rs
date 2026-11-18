//! Native QIF inspection and streaming verification, without a GGUF projection.
use std::{collections::BTreeMap, path::Path};

use quantinfer::cpu::ModelSource;
use quantinfer_format::{
    qif::{FileReader, MetadataValue, PackageReader, TensorInfo},
    ModelManifest,
};
use serde_json::{json, Value};

use super::common::{has_flag, option_value};

pub(crate) fn inspect_gguf(arguments: &[String]) -> Result<(), String> {
    let path = input(arguments)?;
    let source = ModelSource::open(&path).map_err(|error| error.to_string())?;
    let reader = source
        .open_reader(None)
        .map_err(|error| error.to_string())?;
    if has_flag(arguments, "--json") {
        println!(
            "{}",
            json!({"metadata": reader.metadata(), "tensors": reader.tensors()})
        );
        return Ok(());
    }
    println!("metadata: {}", reader.metadata().len());
    for key in reader.metadata().keys() {
        println!("metadata_key={key}");
    }
    for tensor in reader.tensors() {
        println!(
            "{} {:?} {:?} {} bytes",
            tensor.name, tensor.dtype, tensor.shape, tensor.size
        );
    }
    Ok(())
}

fn input(arguments: &[String]) -> Result<String, String> {
    option_value(arguments, "--input")?.ok_or_else(|| "provide --input PATH".into())
}

pub(crate) fn native(arguments: &[String], verify: bool) -> Result<(), String> {
    let input = input(arguments)?;
    let path = Path::new(&input);
    let report = if path.is_dir() {
        if path.join("manifest.json").exists() {
            return Err("native QIF commands require a qif.json package without manifest.json; use inspect-gguf or verify-package for a manifest.json model package".into());
        }
        let mut reader = PackageReader::open(path).map_err(|error| error.to_string())?;
        if verify {
            reader.verify().map_err(|error| error.to_string())?;
        }
        report(
            "PACKAGE",
            reader.manifest(),
            reader.metadata(),
            reader.tensors(),
            verify,
        )
    } else {
        let mut reader = FileReader::open(path).map_err(|error| error.to_string())?;
        if verify {
            reader.verify().map_err(|error| error.to_string())?;
        }
        report(
            "SINGLE",
            reader.manifest(),
            reader.metadata(),
            reader.tensors(),
            verify,
        )
    };
    if has_flag(arguments, "--json") {
        println!("{report}");
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
        );
    }
    Ok(())
}

fn report(
    profile: &str,
    manifest: &ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensors: &[TensorInfo],
    verified: bool,
) -> Value {
    let mut result = json!({
        "format": "QIF", "profile": profile, "model_id": manifest.model_id,
        "control_verified": true, "payload_verified": verified,
        "tensor_count": tensors.len(),
        // Embedded signatures establish mathematical validity, not publisher trust.
        "publisher_trust": "not_evaluated",
    });
    if !verified {
        result["manifest"] = json!(manifest);
        result["metadata"] = json!(metadata);
        result["tensors"] = Value::Array(tensors.iter().map(|tensor| json!({
            "tensor_id": tensor.tensor_id, "name": tensor.name, "shape": tensor.shape,
            "logical_dtype": tensor.dtype, "physical_encoding": tensor.physical_encoding,
            "quantization_scheme": tensor.quantization_scheme, "layout": tensor.layout,
            "codec": tensor.codec, "raw_length": tensor.raw_length,
            "stored_length": tensor.stored_length, "alias_of": tensor.alias_of,
            "checksum_sha256": tensor.checksum_sha256.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        })).collect());
    }
    result
}
