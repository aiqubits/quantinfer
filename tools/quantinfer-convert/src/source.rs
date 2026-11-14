//! SafeTensors input source and checksum validation.

use std::fs;
use std::path::Path;

use quantinfer_format::{sha256_reader, SafeTensorFileReader, SafeTensorShards};

use crate::common::load_checksums;

pub(crate) enum WeightSource {
    Single(SafeTensorFileReader),
    Sharded(SafeTensorShards),
}

impl WeightSource {
    pub(crate) fn open(path: &Path, checksums_path: Option<&Path>) -> Result<Self, String> {
        let checksums = checksums_path.map(load_checksums).transpose()?;
        if let Some(checksums) = &checksums {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("invalid SafeTensors path {}", path.display()))?;
            let expected = checksums
                .get(name)
                .ok_or_else(|| format!("missing checksum for {name}"))?;
            let actual = fs::File::open(path)
                .and_then(|mut file| sha256_reader(&mut file))
                .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
            if &actual != expected {
                return Err(format!(
                    "checksum mismatch for {}: expected {expected}, got {actual}",
                    path.display()
                ));
            }
        }
        if path.to_string_lossy().ends_with(".index.json") {
            let shards = SafeTensorShards::open(path)
                .map_err(|error| format!("cannot parse SafeTensors index: {error}"))?;
            if let Some(checksums) = &checksums {
                shards
                    .verify_checksums(checksums)
                    .map_err(|error| error.to_string())?;
            }
            Ok(Self::Sharded(shards))
        } else {
            SafeTensorFileReader::open(path)
                .map(Self::Single)
                .map_err(|error| format!("cannot parse SafeTensors: {error}"))
        }
    }

    pub(crate) fn tensors(&self) -> Result<Vec<(String, Vec<usize>)>, String> {
        match self {
            Self::Single(file) => Ok(file
                .tensors()
                .iter()
                .map(|(name, info)| (name.clone(), info.shape.clone()))
                .collect()),
            Self::Sharded(files) => Ok(files
                .tensors()
                .map(|(name, info)| (name.to_owned(), info.shape.clone()))
                .collect()),
        }
    }

    pub(crate) fn tensor_f32(&self, name: &str) -> quantinfer_format::Result<Vec<f32>> {
        match self {
            Self::Single(file) => file.tensor_f32(name),
            Self::Sharded(files) => files.tensor_f32(name),
        }
    }

    pub(crate) fn tensor_count(&self) -> usize {
        match self {
            Self::Single(file) => file.tensors().len(),
            Self::Sharded(files) => files.tensors().count(),
        }
    }
}
