//! File-backed SafeTensors shard indexes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::file::{open_readonly_nofollow, SafeTensorFileReader};
use super::types::SafeTensorInfo;
use super::types::MAX_HEADER_BYTES;
use crate::{read_file_nofollow, sha256_reader, FormatError, Result};

#[derive(Debug, Clone)]
pub struct SafeTensorShards {
    index_path: PathBuf,
    tensor_to_shard: BTreeMap<String, String>,
    shards: BTreeMap<String, SafeTensorFileReader>,
}

impl SafeTensorShards {
    pub fn open(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();
        let index_size = fs::metadata(&index_path)
            .map_err(|error| {
                FormatError::Io(format!("cannot stat {}: {error}", index_path.display()))
            })?
            .len();
        if index_size > MAX_HEADER_BYTES as u64 {
            return Err(FormatError::Invalid(
                "SafeTensors index exceeds the size limit".to_owned(),
            ));
        }
        let index_bytes = read_file_nofollow(&index_path).map_err(|error| {
            FormatError::Io(format!("cannot read {}: {error}", index_path.display()))
        })?;
        let value: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|error| {
            FormatError::Invalid(format!(
                "invalid SafeTensors index {}: {error}",
                index_path.display()
            ))
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors index is missing weight_map".to_owned())
            })?;
        let mut tensor_to_shard = BTreeMap::new();
        for (tensor, shard) in weight_map {
            let shard = shard.as_str().ok_or_else(|| {
                FormatError::Invalid(format!("SafeTensors shard for {tensor} is not a string"))
            })?;
            tensor_to_shard.insert(tensor.clone(), shard.to_owned());
        }
        let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
        let mut shards = BTreeMap::new();
        for shard in tensor_to_shard.values() {
            if shards.contains_key(shard) {
                continue;
            }
            let path = resolve_shard_path(parent, shard)?;
            let file = SafeTensorFileReader::open(&path).map_err(|error| {
                FormatError::Io(format!(
                    "cannot open SafeTensors shard {}: {error}",
                    path.display()
                ))
            })?;
            shards.insert(shard.clone(), file);
        }
        for (tensor, shard) in &tensor_to_shard {
            let file = shards.get(shard).ok_or_else(|| {
                FormatError::Invalid(format!("missing loaded SafeTensors shard {shard}"))
            })?;
            if !file.tensors().contains_key(tensor) {
                return Err(FormatError::Invalid(format!(
                    "tensor {tensor} is not present in indexed shard {shard}"
                )));
            }
        }
        Ok(Self {
            index_path,
            tensor_to_shard,
            shards,
        })
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &SafeTensorInfo)> {
        self.tensor_to_shard.iter().filter_map(|(name, shard)| {
            self.shards
                .get(shard)
                .and_then(|file| file.tensors().get(name))
                .map(|info| (name.as_str(), info))
        })
    }

    pub fn tensor_shard(&self, name: &str) -> Result<&str> {
        self.tensor_to_shard
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_f32(name)
    }

    pub fn tensor_u32(&self, name: &str) -> Result<Vec<u32>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_u32(name)
    }

    pub fn tensor_u64(&self, name: &str) -> Result<Vec<u64>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_u64(name)
    }

    pub fn verify_shard_checksum(&self, shard: &str, expected_sha256: &str) -> Result<()> {
        let parent = self.index_path.parent().unwrap_or_else(|| Path::new("."));
        if !self.shards.contains_key(shard) {
            return Err(FormatError::Invalid(format!(
                "unknown SafeTensors shard {shard}"
            )));
        }
        let path = resolve_shard_path(parent, shard)?;
        let mut file = open_readonly_nofollow(&path)
            .map_err(|error| FormatError::Io(format!("cannot read shard {shard}: {error}")))?;
        let actual = sha256_reader(&mut file)
            .map_err(|error| FormatError::Io(format!("cannot hash shard {shard}: {error}")))?;
        if actual != expected_sha256 {
            return Err(FormatError::Invalid(format!(
                "checksum mismatch for shard {shard}: expected {expected_sha256}, got {actual}"
            )));
        }
        Ok(())
    }

    pub fn verify_checksums(&self, expected: &BTreeMap<String, String>) -> Result<()> {
        for shard in self.shards.keys() {
            let checksum = expected.get(shard).ok_or_else(|| {
                FormatError::Invalid(format!("missing checksum for shard {shard}"))
            })?;
            self.verify_shard_checksum(shard, checksum)?;
        }
        Ok(())
    }
}

fn resolve_shard_path(parent: &Path, shard: &str) -> Result<PathBuf> {
    let relative = Path::new(shard);
    let mut has_normal = false;
    for component in relative.components() {
        match component {
            std::path::Component::Normal(_) => has_normal = true,
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(FormatError::Invalid(format!(
                    "invalid SafeTensors shard path {shard}"
                )))
            }
        }
    }
    if !has_normal {
        return Err(FormatError::Invalid(format!(
            "invalid SafeTensors shard path {shard}"
        )));
    }
    let parent = fs::canonicalize(parent).map_err(|error| {
        FormatError::Io(format!(
            "cannot resolve SafeTensors shard directory: {error}"
        ))
    })?;
    let path = fs::canonicalize(parent.join(relative)).map_err(|error| {
        FormatError::Io(format!("cannot resolve SafeTensors shard {shard}: {error}"))
    })?;
    if !path.starts_with(&parent) {
        return Err(FormatError::Invalid(format!(
            "SafeTensors shard {shard} escapes its index directory"
        )));
    }
    Ok(path)
}
