use std::io::Cursor;

use quantinfer_core::{DType, QuantizationScheme, Shape, TensorStorageDescriptor};

use super::{sha256_hex, sha256_reader, ModelManifest, ModelPackage, TensorManifest};

#[test]
fn round_trips_manifest_and_verifies_checksum() {
    let mut manifest = ModelManifest::new("fixture", "qwen3");
    manifest.tensors.push(TensorManifest {
        name: "token_embd.weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2, 2])).unwrap(),
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
fn hashes_known_bytes_and_streams_identically() {
    let expected = "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d";
    assert_eq!(sha256_hex(b"fixture"), expected);
    assert_eq!(
        sha256_reader(&mut Cursor::new(b"fixture")).unwrap(),
        expected
    );
}

#[test]
fn accepts_uppercase_sha256_checksums() {
    let mut manifest = ModelManifest::new("fixture", "qwen3");
    let bytes = b"model";
    manifest.insert_checksum("model.gguf", bytes);
    let checksum = manifest.checksums.get_mut("model.gguf").unwrap();
    *checksum = checksum.to_ascii_uppercase();
    manifest.validate().unwrap();
    manifest.verify_checksum("model.gguf", bytes).unwrap();
}

#[test]
fn rejects_malformed_checksums_and_duplicate_variants() {
    let mut manifest = ModelManifest::new("fixture", "qwen3");
    manifest.insert_checksum("model.gguf", b"fixture");
    manifest
        .checksums
        .insert("../escape".to_owned(), "00".to_owned());
    assert!(manifest.to_json().is_err());

    let mut manifest = ModelManifest::new("fixture", "qwen3");
    manifest.execution_variants.extend([
        super::ExecutionVariant {
            name: "cpu".to_owned(),
            quantization: QuantizationScheme::None,
            dtype: "f32".to_owned(),
            runtime_provider: "cpu".to_owned(),
            kernel_provider: "cpu".to_owned(),
        },
        super::ExecutionVariant {
            name: "cpu".to_owned(),
            quantization: QuantizationScheme::None,
            dtype: "f32".to_owned(),
            runtime_provider: "cpu".to_owned(),
            kernel_provider: "cpu".to_owned(),
        },
    ]);
    assert!(manifest.to_json().is_err());
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
    let outside = root.with_file_name(format!("quantinfer-package-outside-{}", std::process::id()));
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

#[cfg(unix)]
#[test]
fn rejects_manifests_that_escape_through_symlinks() {
    let root = std::env::temp_dir().join(format!(
        "quantinfer-manifest-symlink-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let outside = root.with_file_name(format!(
        "quantinfer-manifest-outside-{}",
        std::process::id()
    ));
    std::fs::write(
        &outside,
        ModelManifest::new("outside", "qwen3").to_json().unwrap(),
    )
    .unwrap();
    std::os::unix::fs::symlink(&outside, root.join("manifest.json")).unwrap();
    let error = ModelPackage::open(&root).unwrap_err().to_string();
    assert!(error.contains("outside the package root"));
    std::fs::remove_file(root.join("manifest.json")).unwrap();
    std::fs::remove_file(outside).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
