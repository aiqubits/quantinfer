//! Immutable, bounded JSONL calibration input for offline model workflows.

use std::{fs::File, io::Read, path::Path};

use quantinfer_format::sha256_hex;

pub(super) struct PromptDataset {
    pub texts: Vec<String>,
    pub sha256: String,
}

pub(super) fn read_dataset(path: &Path, maximum_bytes: usize) -> Result<PromptDataset, String> {
    let limit = maximum_bytes
        .checked_add(1)
        .ok_or("dataset size limit overflows")?;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))?
        .take(limit as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if bytes.len() > maximum_bytes {
        return Err(format!(
            "{} exceeds the dataset byte budget",
            path.display()
        ));
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|error| format!("dataset is not UTF-8: {error}"))?;
    let mut texts = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid JSON at dataset line {}: {error}", index + 1))?;
        let prompt = value
            .as_str()
            .or_else(|| value.get("text").and_then(serde_json::Value::as_str))
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("dataset line {} requires a nonempty text string", index + 1))?;
        texts.push(prompt.to_owned());
    }
    if texts.is_empty() {
        return Err("dataset contains no prompts".into());
    }
    Ok(PromptDataset {
        texts,
        sha256: sha256_hex(&bytes),
    })
}
