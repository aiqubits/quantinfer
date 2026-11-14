//! GGUF to QIF transcoding workflow.

use std::fs;
use std::path::Path;

use quantinfer_format::{qif::import_gguf, GgufReader};

use crate::common::{optional_value, required_path};

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
