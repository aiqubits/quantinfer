//! Verified model package filesystem access.

use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use super::hash::{
    open_readonly_nofollow, read_file_nofollow, sha256_hex, sha256_reader, verify_digest,
};
use super::types::ModelManifest;
use crate::{FormatError, Result};

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
        let manifest_resolved = fs::canonicalize(&manifest_path).map_err(|error| {
            FormatError::Io(format!("cannot read {}: {error}", manifest_path.display()))
        })?;
        if !manifest_resolved.starts_with(&root) {
            return Err(FormatError::Invalid(
                "manifest.json resolves outside the package root".to_owned(),
            ));
        }
        let manifest_bytes = read_file_nofollow(&manifest_resolved).map_err(|error| {
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
