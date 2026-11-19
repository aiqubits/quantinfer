//! GGUF to QIF transcoding workflow.

use std::fs;

use quantinfer_format::GgufReader;

use crate::common::{optional_value, required_path};

pub(crate) fn convert_qif(arguments: &[String]) -> Result<(), String> {
    let input = required_path(arguments, "--input")?;
    let output = required_path(arguments, "--output")?;
    let options = super::qif_options::QifOptions::parse(arguments)?;
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
    options.publish(&reader, &model_id, &output)
}

pub(crate) fn repack_qif(arguments: &[String]) -> Result<(), String> {
    let input = required_path(arguments, "--input")?;
    let output = required_path(arguments, "--output")?;
    let options = super::qif_options::QifOptions::parse_with_default(arguments, true)?;
    let reader = GgufReader::open_path(input).map_err(|error| error.to_string())?;
    options.repack(&reader, &output)
}
