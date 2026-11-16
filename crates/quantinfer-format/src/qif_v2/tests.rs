use std::collections::BTreeMap;

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};

use super::{write_qif_v2, QifV2FileReader, QifV2MappedFile, QifV2MetadataValue, QifV2Reader};
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
