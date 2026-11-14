//! Model manifest schema and validation.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use quantinfer_core::{QuantizationScheme, TensorStorageDescriptor};
use serde::{Deserialize, Serialize};

use super::hash::{sha256_hex, verify_digest};
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
            if let Some(checksum) = &tensor.checksum_sha256 {
                if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(FormatError::Invalid(format!(
                        "invalid SHA-256 checksum for tensor {}",
                        tensor.name
                    )));
                }
            }
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
            if variant.quantization != self.quantization {
                return Err(FormatError::Invalid(format!(
                    "execution variant {} has quantization {} but manifest declares {}",
                    variant.name,
                    variant.quantization.name(),
                    self.quantization.name()
                )));
            }
        }
        if self
            .execution_variants
            .iter()
            .map(|variant| variant.name.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != self.execution_variants.len()
        {
            return Err(FormatError::Invalid(
                "execution variant names must be unique".to_owned(),
            ));
        }
        for (path, checksum) in &self.checksums {
            let components = Path::new(path).components();
            if path.is_empty()
                || components.clone().any(|part| {
                    matches!(
                        part,
                        Component::Prefix(_) | Component::RootDir | Component::ParentDir
                    )
                })
                || !components
                    .clone()
                    .any(|part| matches!(part, Component::Normal(_)))
                || checksum.len() != 64
                || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(FormatError::Invalid(format!(
                    "invalid SHA-256 checksum entry for {path}"
                )));
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
        verify_digest(path, self.expected_checksum(path)?, &sha256_hex(bytes))
    }

    pub(crate) fn expected_checksum(&self, path: &str) -> Result<&str> {
        self.checksums
            .get(path)
            .map(String::as_str)
            .ok_or_else(|| FormatError::Invalid(format!("manifest has no checksum for {path}")))
    }

    pub fn checksums_sha256(&self) -> String {
        self.checksums
            .iter()
            .map(|(path, checksum)| format!("{checksum}  {path}"))
            .collect::<Vec<_>>()
            .join("\n")
            + if self.checksums.is_empty() { "" } else { "\n" }
    }
}
