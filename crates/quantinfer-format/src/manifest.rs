use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};

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
        verify_digest(path, self.expected_checksum(path)?, &sha256_hex(bytes))
    }

    fn expected_checksum(&self, path: &str) -> Result<&str> {
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

#[derive(Debug, Clone)]
pub struct ModelPackage {
    root: PathBuf,
    pub manifest: ModelManifest,
}

impl ModelPackage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref()).map_err(|error| {
            FormatError::Io(format!(
                "cannot resolve package root {}: {error}",
                root.as_ref().display()
            ))
        })?;
        let manifest_path = root.join("manifest.json");
        let manifest_bytes = read_file_nofollow(&manifest_path).map_err(|error| {
            FormatError::Io(format!("cannot read {}: {error}", manifest_path.display()))
        })?;
        let manifest_text = String::from_utf8(manifest_bytes).map_err(|error| {
            FormatError::Invalid(format!(
                "manifest {} is not UTF-8: {error}",
                manifest_path.display()
            ))
        })?;
        let manifest = ModelManifest::from_json(&manifest_text)?;
        Ok(Self { root, manifest })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a package-relative artifact name.
    pub fn artifact_path(&self, relative_path: &str) -> Result<PathBuf> {
        let path = Path::new(relative_path);
        let components = path.components();
        if !components
            .clone()
            .any(|part| matches!(part, Component::Normal(_)))
            || components.clone().any(|part| {
                matches!(
                    part,
                    Component::Prefix(_) | Component::RootDir | Component::ParentDir
                )
            })
        {
            return Err(FormatError::Invalid(format!(
                "invalid package artifact path {relative_path}"
            )));
        }
        let path = self.root.join(path);
        let resolved = fs::canonicalize(&path).map_err(|error| {
            FormatError::Io(format!(
                "cannot resolve package artifact {}: {error}",
                path.display()
            ))
        })?;
        if !resolved.starts_with(&self.root) {
            return Err(FormatError::Invalid(format!(
                "package artifact {relative_path} resolves outside the package root"
            )));
        }
        Ok(resolved)
    }

    pub fn read_verified(&self, relative_path: &str) -> Result<Vec<u8>> {
        let path = self.artifact_path(relative_path)?;
        let expected = self.manifest.expected_checksum(relative_path)?;
        let mut file = open_readonly_nofollow(&path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        verify_digest(relative_path, expected, &sha256_hex(&bytes))?;
        Ok(bytes)
    }

    /// Verify every declared file by streaming its SHA-256 digest.
    pub fn verify_all(&self) -> Result<usize> {
        if self.manifest.checksums.is_empty() {
            return Err(FormatError::Invalid(
                "package manifest declares no artifact checksums".to_owned(),
            ));
        }
        for (name, expected) in &self.manifest.checksums {
            let path = self.artifact_path(name)?;
            let actual = open_readonly_nofollow(&path)
                .and_then(|mut file| sha256_reader(&mut file))
                .map_err(|error| {
                    FormatError::Io(format!("cannot hash {}: {error}", path.display()))
                })?;
            verify_digest(name, expected, &actual)?;
        }
        Ok(self.manifest.checksums.len())
    }
}

fn open_readonly_nofollow(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    options.open(path)
}

fn verify_digest(path: &str, expected: &str, actual: &str) -> Result<()> {
    if expected != actual {
        return Err(FormatError::Invalid(format!(
            "checksum mismatch for {path}: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_reader(reader: &mut impl Read) -> io::Result<String> {
    let mut digest = Sha256::new();
    io::copy(reader, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

/// Resolve a path once, then read the resolved regular file without following
/// a replacement final symlink on Unix.
pub fn read_file_nofollow(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let resolved = fs::canonicalize(path.as_ref())?;
    let mut file = open_readonly_nofollow(&resolved)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
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
        assert!(decoded.checksums_sha256().ends_with("  model.gguf\n"));
    }

    #[test]
    fn opens_a_directory_package_and_reads_verified_model_bytes() {
        let root = std::env::temp_dir().join(format!("quantinfer-package-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let model_bytes = b"fixture-gguf";
        std::fs::write(root.join("model.gguf"), model_bytes).unwrap();
        std::fs::create_dir_all(root.join("profiles")).unwrap();
        let profile_path = root.join("profiles/default.json");
        std::fs::write(&profile_path, b"fixture-profile").unwrap();
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        manifest.insert_checksum("model.gguf", model_bytes);
        manifest.insert_checksum("profiles/default.json", b"fixture-profile");
        std::fs::write(root.join("manifest.json"), manifest.to_json().unwrap()).unwrap();
        let package = ModelPackage::open(&root).unwrap();
        assert_eq!(package.read_verified("model.gguf").unwrap(), model_bytes);
        assert_eq!(package.verify_all().unwrap(), 2);
        assert!(package.read_verified("../model.gguf").is_err());
        std::fs::write(&profile_path, b"changed-profile").unwrap();
        assert!(package
            .verify_all()
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch for profiles/default.json"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_package_artifacts_that_escape_through_symlinks() {
        let root =
            std::env::temp_dir().join(format!("quantinfer-package-symlink-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let outside =
            root.with_file_name(format!("quantinfer-package-outside-{}", std::process::id()));
        std::fs::write(&outside, b"outside").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("model.gguf")).unwrap();
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        manifest.insert_checksum("model.gguf", b"outside");
        std::fs::write(root.join("manifest.json"), manifest.to_json().unwrap()).unwrap();
        let package = ModelPackage::open(&root).unwrap();
        let error = package.verify_all().unwrap_err().to_string();
        assert!(error.contains("outside the package root"));
        std::fs::remove_file(root.join("model.gguf")).unwrap();
        std::fs::remove_file(outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
