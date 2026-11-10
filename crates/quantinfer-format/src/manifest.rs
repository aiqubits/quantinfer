use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quantinfer_core::{QuantizationScheme, TensorStorageDescriptor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{FormatError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: String,
    pub descriptor: TensorStorageDescriptor,
    pub checksum_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionVariant {
    pub name: String,
    pub quantization: QuantizationScheme,
    pub dtype: String,
    pub runtime_provider: String,
    pub kernel_provider: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KernelRequirements {
    pub operations: Vec<String>,
    pub alignment: Option<usize>,
    pub head_dimension: Option<usize>,
    pub maximum_sequence_length: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Compatibility {
    pub minimum_quantinfer_version: Option<String>,
    pub gguf_version: Option<u32>,
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub architecture: String,
    pub source_revision: Option<String>,
    pub source_format: String,
    pub tokenizer_revision: Option<String>,
    pub tensors: Vec<TensorManifest>,
    pub quantization: QuantizationScheme,
    pub execution_variants: Vec<ExecutionVariant>,
    pub kernel_requirements: KernelRequirements,
    pub checksums: BTreeMap<String, String>,
    pub compatibility: Compatibility,
}

impl ModelManifest {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    pub fn new(model_id: impl Into<String>, architecture: impl Into<String>) -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            model_id: model_id.into(),
            architecture: architecture.into(),
            source_revision: None,
            source_format: "gguf".to_owned(),
            tokenizer_revision: None,
            tensors: Vec::new(),
            quantization: QuantizationScheme::None,
            execution_variants: Vec::new(),
            kernel_requirements: KernelRequirements::default(),
            checksums: BTreeMap::new(),
            compatibility: Compatibility::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version == 0 || self.schema_version > Self::CURRENT_SCHEMA_VERSION {
            return Err(FormatError::Invalid(format!(
                "unsupported model manifest schema version {}",
                self.schema_version
            )));
        }
        if self.model_id.is_empty() || self.architecture.is_empty() || self.source_format.is_empty()
        {
            return Err(FormatError::Invalid(
                "model manifest identity fields must not be empty".to_owned(),
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        for tensor in &self.tensors {
            if tensor.name.is_empty() || !names.insert(&tensor.name) {
                return Err(FormatError::Invalid(
                    "model manifest tensor names must be unique and non-empty".to_owned(),
                ));
            }
            tensor.descriptor.validate()?;
        }
        for variant in &self.execution_variants {
            if variant.name.is_empty()
                || variant.runtime_provider.is_empty()
                || variant.kernel_provider.is_empty()
            {
                return Err(FormatError::Invalid(
                    "execution variant identity fields must not be empty".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| {
            FormatError::Invalid(format!("cannot serialize model manifest: {error}"))
        })
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(json).map_err(|error| {
            FormatError::Invalid(format!("cannot parse model manifest: {error}"))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn insert_checksum(&mut self, path: impl Into<String>, bytes: &[u8]) {
        self.checksums.insert(path.into(), sha256_hex(bytes));
    }

    pub fn verify_checksum(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let expected = self
            .checksums
            .get(path)
            .ok_or_else(|| FormatError::Invalid(format!("manifest has no checksum for {path}")))?;
        let actual = sha256_hex(bytes);
        if expected != &actual {
            return Err(FormatError::Invalid(format!(
                "checksum mismatch for {path}: expected {expected}, got {actual}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ModelPackage {
    root: PathBuf,
    pub manifest: ModelManifest,
}

impl ModelPackage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join("manifest.json");
        let manifest =
            ModelManifest::from_json(&fs::read_to_string(&manifest_path).map_err(|error| {
                FormatError::Io(format!("cannot read {}: {error}", manifest_path.display()))
            })?)?;
        Ok(Self { root, manifest })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn read_verified(&self, relative_path: &str) -> Result<Vec<u8>> {
        let path = self.root.join(relative_path);
        let bytes = fs::read(&path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        self.manifest.verify_checksum(relative_path, &bytes)?;
        Ok(bytes)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use quantinfer_core::{DType, QuantizationScheme, Shape, TensorStorageDescriptor};

    use super::{sha256_hex, ModelManifest, ModelPackage, TensorManifest};

    #[test]
    fn round_trips_manifest_and_verifies_checksum() {
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        manifest.tensors.push(TensorManifest {
            name: "token_embd.weight".to_owned(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2, 2]))
                .unwrap(),
            checksum_sha256: None,
        });
        manifest.insert_checksum("model.gguf", b"fixture");
        let json = manifest.to_json().unwrap();
        let decoded = ModelManifest::from_json(&json).unwrap();
        decoded.verify_checksum("model.gguf", b"fixture").unwrap();
        assert_eq!(decoded.quantization, QuantizationScheme::None);
        assert_eq!(sha256_hex(b"fixture").len(), 64);
    }

    #[test]
    fn opens_a_directory_package_and_reads_verified_model_bytes() {
        let root = std::env::temp_dir().join(format!("quantinfer-package-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let model_bytes = b"fixture-gguf";
        std::fs::write(root.join("model.gguf"), model_bytes).unwrap();
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        manifest.insert_checksum("model.gguf", model_bytes);
        std::fs::write(root.join("manifest.json"), manifest.to_json().unwrap()).unwrap();
        let package = ModelPackage::open(&root).unwrap();
        assert_eq!(package.read_verified("model.gguf").unwrap(), model_bytes);
        std::fs::remove_dir_all(root).unwrap();
    }
}
