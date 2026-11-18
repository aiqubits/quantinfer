//! Runtime preflight for mandatory manifest declarations.
use quantinfer_format::ModelManifest;
use quantinfer_kernel::{BackendId, KernelBackend};
use semver::Version;

use crate::{Operation, Result, RuntimeError};

fn invalid(message: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidInput(message.into())
}

/// Format capabilities present in every build of the current runtime. These
/// refer to parsing/decoding semantics, not a particular filesystem transport.
pub const RUNTIME_FEATURES: &[&str] = &[
    "qif.metadata.map",
    "qif.tensor.alias",
    "qif.codec.zstd",
    "qif.codec.lz4",
];

/// Validate version and registered names without selecting a backend. Actual
/// model execution must additionally call `validate_manifest_for_backend`.
pub fn validate_manifest_for_runtime(manifest: &ModelManifest) -> Result<()> {
    manifest.validate()?;
    if let Some(minimum) = &manifest.compatibility.minimum_quantinfer_version {
        let required = Version::parse(minimum)
            .map_err(|_| invalid("minimum_quantinfer_version must be a full SemVer version"))?;
        let current = Version::parse(env!("CARGO_PKG_VERSION"))
            .map_err(|_| invalid("invalid compiled runtime version"))?;
        if current.cmp_precedence(&required).is_lt() {
            return Err(invalid(format!(
                "model requires QuantInfer {minimum}, current runtime is {current}"
            )));
        }
    }
    for feature in &manifest.compatibility.required_features {
        if !RUNTIME_FEATURES.contains(&feature.as_str()) {
            return Err(invalid(format!("unsupported required feature {feature}")));
        }
    }
    for tensor in &manifest.tensors {
        for requirement in &tensor.descriptor.backend_requirements {
            if BackendId::from_name(requirement).is_none() {
                return Err(invalid(format!(
                    "unsupported backend requirement {requirement} for tensor {}",
                    tensor.name
                )));
            }
        }
    }
    let requirements = &manifest.kernel_requirements;
    if requirements.alignment == Some(0)
        || requirements.head_dimension == Some(0)
        || requirements.maximum_sequence_length == Some(0)
    {
        return Err(invalid(
            "kernel requirements must be positive when specified",
        ));
    }
    required_manifest_operations(manifest)?;
    Ok(())
}

pub fn validate_manifest_for_backend(
    manifest: &ModelManifest,
    backend: &(impl KernelBackend + ?Sized),
) -> Result<()> {
    validate_manifest_for_backend_id(manifest, backend.backend_id())
}

/// Separate identity check for capability planning before a GPU is initialized.
pub fn validate_manifest_for_backend_id(
    manifest: &ModelManifest,
    backend: Option<BackendId>,
) -> Result<()> {
    validate_manifest_for_runtime(manifest)?;
    for tensor in &manifest.tensors {
        for name in &tensor.descriptor.backend_requirements {
            if BackendId::from_name(name) != backend {
                return Err(invalid(format!(
                    "backend requirement {name} for tensor {} is not satisfied by {}",
                    tensor.name,
                    backend.map_or("unregistered backend", BackendId::name)
                )));
            }
        }
    }
    Ok(())
}

/// Stable operation spellings accepted by the current model manifest runtime.
pub fn required_manifest_operations(manifest: &ModelManifest) -> Result<Vec<Operation>> {
    manifest
        .kernel_requirements
        .operations
        .iter()
        .map(|name| match name.as_str() {
            "matmul" => Ok(Operation::Matmul),
            "quantized_matmul" => Ok(Operation::QuantizedMatmul),
            "attention" => Ok(Operation::Attention),
            "rms_norm" => Ok(Operation::RmsNorm),
            "activation" => Ok(Operation::Activation),
            "rope" => Ok(Operation::RoPE),
            "embedding" => Ok(Operation::Embedding),
            _ => Err(invalid(format!("unknown required operation {name}"))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
    use quantinfer_format::TensorManifest;

    fn manifest() -> ModelManifest {
        let mut manifest = ModelManifest::new("requirements", "llama");
        manifest.tensors.push(TensorManifest {
            name: "weight".into(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))
                .unwrap(),
            checksum_sha256: None,
        });
        manifest
    }

    #[test]
    fn registered_features_and_backend_names_match_the_selected_implementation() {
        let mut manifest = manifest();
        manifest.compatibility.required_features =
            RUNTIME_FEATURES.iter().map(|name| (*name).into()).collect();
        validate_manifest_for_runtime(&manifest).unwrap();
        let mut names = std::collections::BTreeSet::new();
        for &backend in BackendId::ALL {
            assert!(names.insert(backend.name()));
            assert_eq!(BackendId::from_name(backend.name()), Some(backend));
            manifest.tensors[0].descriptor.backend_requirements = vec![backend.name().into()];
            validate_manifest_for_runtime(&manifest).unwrap();
            validate_manifest_for_backend_id(&manifest, Some(backend)).unwrap();
            assert!(validate_manifest_for_backend_id(&manifest, None).is_err());
            for &other in BackendId::ALL {
                assert_eq!(
                    validate_manifest_for_backend_id(&manifest, Some(other)).is_ok(),
                    backend == other
                );
            }
        }
        manifest.tensors[0].descriptor.backend_requirements = vec!["cpu".into(), "cuda".into()];
        assert!(validate_manifest_for_backend_id(&manifest, Some(BackendId::Cpu)).is_err());
        manifest.tensors[0].descriptor.backend_requirements.clear();
        manifest
            .compatibility
            .required_features
            .push("qif.codec.unknown".into());
        assert!(validate_manifest_for_runtime(&manifest).is_err());
    }

    #[test]
    fn version_precedence_is_semantic_and_ignores_build_metadata() {
        let mut manifest = manifest();
        for version in [
            env!("CARGO_PKG_VERSION"),
            "0.0.1",
            "0.1.0-alpha.1",
            "0.1.0+build.1",
        ] {
            manifest.compatibility.minimum_quantinfer_version = Some(version.into());
            validate_manifest_for_runtime(&manifest).unwrap();
        }
        for version in ["999.0.0", "1.0.0", "invalid", "0.1", ""] {
            manifest.compatibility.minimum_quantinfer_version = Some(version.into());
            assert!(
                validate_manifest_for_runtime(&manifest).is_err(),
                "{version}"
            );
        }
    }

    #[test]
    fn unknown_mandatory_features_backends_and_operations_cannot_be_ignored() {
        let mut manifest = manifest();
        manifest
            .compatibility
            .required_features
            .push("future-feature".into());
        assert!(validate_manifest_for_runtime(&manifest)
            .unwrap_err()
            .to_string()
            .contains("required feature"));
        manifest.compatibility.required_features.clear();
        manifest.tensors[0]
            .descriptor
            .backend_requirements
            .push("future-backend".into());
        assert!(validate_manifest_for_runtime(&manifest)
            .unwrap_err()
            .to_string()
            .contains("backend requirement"));
        manifest.tensors[0].descriptor.backend_requirements.clear();
        manifest
            .kernel_requirements
            .operations
            .push("future-operation".into());
        assert!(validate_manifest_for_runtime(&manifest)
            .unwrap_err()
            .to_string()
            .contains("required operation"));
        manifest.kernel_requirements.operations = [
            "matmul",
            "quantized_matmul",
            "attention",
            "rms_norm",
            "activation",
            "rope",
            "embedding",
        ]
        .map(str::to_owned)
        .to_vec();
        assert_eq!(required_manifest_operations(&manifest).unwrap().len(), 7);
        validate_manifest_for_runtime(&manifest).unwrap();
        manifest.kernel_requirements.alignment = Some(0);
        assert!(validate_manifest_for_runtime(&manifest).is_err());
    }
}
