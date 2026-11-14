//! Shared converter parsing, checksum, and imatrix helpers.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quantinfer_format::{is_linear_projection_storage_name, GgufReader, GgufValue};

pub(crate) fn load_checksums(path: &Path) -> Result<BTreeMap<String, String>, String> {
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

pub(crate) fn required_path(arguments: &[String], flag: &str) -> Result<PathBuf, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {flag}; use --help for usage"))
}

pub(crate) fn optional_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

pub(crate) fn metadata_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} does not fit in GGUF u32 metadata"))
}

pub(crate) fn is_quantizable_linear(name: &str) -> bool {
    is_linear_projection_storage_name(name)
}

pub(crate) fn load_imatrix(bytes: Vec<u8>) -> Result<BTreeMap<String, Vec<f32>>, String> {
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
