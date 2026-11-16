use std::collections::BTreeMap;

use ed25519_dalek::SigningKey;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};

use super::{
    write_qif_v2, write_qif_v2_package_with_codec, write_qif_v2_signed, write_qif_v2_with_codec,
    QifV2Codec, QifV2FileReader, QifV2MappedFile, QifV2MetadataValue, QifV2PackageReader,
    QifV2Reader,
};
use crate::{ModelManifest, TensorManifest};

fn sample_manifest() -> ModelManifest {
    let mut manifest = ModelManifest::new("test-model", "test");
    manifest.source_format = "qif-v2-test".to_owned();
    manifest.tensors.push(TensorManifest {
        name: "weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });
    manifest
}

#[test]
fn round_trips_single_file_and_packed_adapter() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_owned(),
        QifV2MetadataValue::String("test".to_owned()),
    );
    metadata.insert(
        "quantinfer.qif.ggml_dimension_order".to_owned(),
        QifV2MetadataValue::Bool(false),
    );
    let mut bytes = Vec::new();
    write_qif_v2(sample_manifest(), &metadata, |_| Ok(&payload), &mut bytes).unwrap();

    let reader = QifV2Reader::open(&bytes).unwrap();
    assert_eq!(reader.tensor_bytes("weight").unwrap(), payload);
    assert_eq!(reader.read_range("weight", 4..8).unwrap(), payload[4..]);
    reader.verify().unwrap();

    let packed = crate::GgufReader::open_artifact(bytes).unwrap();
    assert_eq!(packed.tensor_bytes("weight").unwrap(), payload);
    assert_eq!(packed.embedded_manifest().unwrap().model_id, "test-model");
}

#[test]
fn rejects_mixed_metadata_array() {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "bad".to_owned(),
        QifV2MetadataValue::Array(vec![
            QifV2MetadataValue::U32(1),
            QifV2MetadataValue::String("mixed".to_owned()),
        ]),
    );
    let error = super::metadata::encode(&metadata).unwrap_err();
    assert!(error.to_string().contains("one tag"));
}

#[test]
fn rejects_noncanonical_manifest_bytes() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut bytes = Vec::new();
    write_qif_v2(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut bytes,
    )
    .unwrap();
    let manifest_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    bytes[manifest_offset] = b' ';
    assert!(QifV2Reader::open(&bytes).is_err());
}

#[test]
fn file_reader_loads_control_sections_lazily() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut bytes = Vec::new();
    write_qif_v2(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut bytes,
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!(
        "quantinfer-qif-v2-{}-{}.qif",
        std::process::id(),
        payload[0]
    ));
    std::fs::write(&path, bytes).unwrap();
    let mut reader = QifV2FileReader::open(&path).unwrap();
    assert_eq!(reader.read_range("weight", 0..4).unwrap(), payload[..4]);
    reader.verify().unwrap();
    let mapped = QifV2MappedFile::open(&path).unwrap();
    assert_eq!(
        mapped.reader().unwrap().tensor_bytes("weight").unwrap(),
        payload
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn rejects_unknown_flags_and_detects_payload_corruption_on_verify() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut bytes = Vec::new();
    write_qif_v2(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut bytes,
    )
    .unwrap();
    bytes[8] = 1;
    assert!(QifV2Reader::open(&bytes).is_err());

    let mut bytes = Vec::new();
    write_qif_v2(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut bytes,
    )
    .unwrap();
    let payload_offset = u64::from_le_bytes(bytes[72..80].try_into().unwrap()) as usize;
    bytes[payload_offset] ^= 1;
    let reader = QifV2Reader::open(&bytes).unwrap();
    assert!(reader.verify().is_err());
}

#[test]
fn round_trips_independently_compressed_codecs() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    for codec in [QifV2Codec::Zstd, QifV2Codec::Lz4] {
        let mut bytes = Vec::new();
        write_qif_v2_with_codec(
            sample_manifest(),
            &BTreeMap::new(),
            |_| Ok(&payload),
            &mut bytes,
            codec,
        )
        .unwrap();
        let reader = QifV2Reader::open(&bytes).unwrap();
        assert_eq!(reader.tensor_data("weight").unwrap(), payload);
        assert_eq!(reader.read_range("weight", 4..8).unwrap(), payload[4..]);
        reader.verify().unwrap();
        assert!(reader.tensor_bytes("weight").is_err());
        let packed = crate::GgufReader::open_artifact(bytes).unwrap();
        assert_eq!(packed.tensor_bytes("weight").unwrap(), payload);
    }
}

#[test]
fn verifies_ed25519_signature_envelope() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
    let mut bytes = Vec::new();
    write_qif_v2_signed(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut bytes,
        QifV2Codec::None,
        &signing_key,
    )
    .unwrap();
    let reader = QifV2Reader::open(&bytes).unwrap();
    reader.verify().unwrap();
    let signature_offset = u64::from_le_bytes(bytes[120..128].try_into().unwrap()) as usize;
    bytes[signature_offset + 50] ^= 1;
    assert!(QifV2Reader::open(&bytes).is_err());
}

#[test]
fn round_trips_qif2_package_and_shard_ranges() {
    let mut manifest = sample_manifest();
    manifest.tensors.push(TensorManifest {
        name: "bias".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });
    manifest.tensors.push(TensorManifest {
        name: "large".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![4])).unwrap(),
        checksum_sha256: None,
    });
    let payloads = BTreeMap::from([
        ("weight".to_owned(), vec![0_u8, 0, 128, 63, 0, 0, 0, 64]),
        ("bias".to_owned(), vec![0_u8, 0, 0, 0, 0, 0, 128, 63]),
        ("large".to_owned(), vec![0_u8; 16]),
    ]);
    let output =
        std::env::temp_dir().join(format!("quantinfer-qif2-package-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&output);
    write_qif_v2_package_with_codec(
        manifest,
        &BTreeMap::new(),
        |name| {
            payloads
                .get(name)
                .map(Vec::as_slice)
                .ok_or_else(|| crate::FormatError::Invalid("missing package tensor".to_owned()))
        },
        &output,
        Some(8),
        QifV2Codec::Lz4,
    )
    .unwrap();
    let mut reader = QifV2PackageReader::open(&output).unwrap();
    assert_eq!(reader.tensors().len(), 3);
    assert_eq!(reader.tensor_shard("weight").unwrap(), "shard-00000.qifd");
    assert_eq!(reader.tensor_shard("bias").unwrap(), "shard-00001.qifd");
    assert_eq!(
        reader.read_range("bias", 4..8).unwrap(),
        payloads["bias"][4..]
    );
    assert_eq!(
        reader.read_range("large", 0..16).unwrap(),
        payloads["large"]
    );
    reader.verify().unwrap();
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn deduplicates_shared_payloads_in_single_and_package_profiles() {
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut manifest = sample_manifest();
    manifest.tensors.push(TensorManifest {
        name: "weight_alias".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });

    let mut single = Vec::new();
    write_qif_v2(
        manifest.clone(),
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut single,
    )
    .unwrap();
    let reader = QifV2Reader::open(&single).unwrap();
    assert_eq!(
        reader.tensor_info("weight_alias").unwrap().alias_of,
        Some(0)
    );
    assert_eq!(reader.tensor_data("weight_alias").unwrap(), payload);
    reader.verify().unwrap();

    let output = std::env::temp_dir().join(format!(
        "quantinfer-qif2-package-alias-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&output);
    write_qif_v2_package_with_codec(
        manifest,
        &BTreeMap::new(),
        |_| Ok(&payload),
        &output,
        Some(1024),
        QifV2Codec::None,
    )
    .unwrap();
    let mut package = QifV2PackageReader::open(&output).unwrap();
    assert_eq!(
        package.tensor_info("weight_alias").unwrap().alias_of,
        Some(0)
    );
    assert_eq!(
        package.tensor_shard("weight_alias").unwrap(),
        "shard-00000.qifd"
    );
    assert_eq!(
        package
            .read_range("weight_alias", 0..payload.len())
            .unwrap(),
        payload
    );
    package.verify().unwrap();
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn package_accepts_multiple_physical_tensors_in_one_shard() {
    let mut manifest = sample_manifest();
    manifest.tensors.push(TensorManifest {
        name: "bias".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });
    let payloads = BTreeMap::from([
        ("weight".to_owned(), vec![0_u8, 0, 128, 63, 0, 0, 0, 64]),
        ("bias".to_owned(), vec![0_u8, 0, 0, 0, 0, 0, 128, 63]),
    ]);
    let output = std::env::temp_dir().join(format!(
        "quantinfer-qif2-package-one-shard-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&output);
    write_qif_v2_package_with_codec(
        manifest,
        &BTreeMap::new(),
        |name| {
            payloads
                .get(name)
                .map(Vec::as_slice)
                .ok_or_else(|| crate::FormatError::Invalid("missing package tensor".to_owned()))
        },
        &output,
        Some(1024 * 1024),
        QifV2Codec::None,
    )
    .unwrap();
    let mut reader = QifV2PackageReader::open(&output).unwrap();
    assert_eq!(
        reader.read_range("weight", 0..8).unwrap(),
        payloads["weight"]
    );
    assert_eq!(reader.read_range("bias", 0..8).unwrap(), payloads["bias"]);
    reader.verify().unwrap();
    std::fs::remove_dir_all(output).unwrap();
}
