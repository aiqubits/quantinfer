use quantinfer_core::{Q4_0Block, QuantizedTensor4};

use super::*;
use crate::{GgufWriter, QUANT_BLOCK_LAYOUT};

const WEIGHT: [u8; 18] = [
    0, 60, 0x91, 0x82, 0x73, 0x64, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
];
const NORM: [u8; 4] = 1.5f32.to_le_bytes();

fn model() -> (ModelManifest, BTreeMap<String, GgufValue>) {
    let mut manifest = ModelManifest::new("qif-fixture", "llama");
    manifest.quantization = QuantizationScheme::Q4_0;
    manifest.tensors = vec![
        TensorManifest {
            name: "norm".to_owned(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(1)).unwrap(),
            checksum_sha256: None,
        },
        TensorManifest {
            name: "weight".to_owned(),
            descriptor: TensorStorageDescriptor::quantized(
                DType::Q4_0,
                QuantizationScheme::Q4_0,
                Shape::from(vec![1, 32]),
                Q4_0_BLOCK_SIZE,
            )
            .unwrap(),
            checksum_sha256: None,
        },
    ];
    let metadata = BTreeMap::from([
        ("context".to_owned(), GgufValue::U64(4096)),
        ("empty".to_owned(), GgufValue::Array(Vec::new())),
        (
            "tokens".to_owned(),
            GgufValue::Array(vec![GgufValue::String("量化".to_owned())]),
        ),
    ]);
    (manifest, metadata)
}

fn artifact() -> Vec<u8> {
    let (manifest, metadata) = model();
    let mut bytes = Vec::new();
    write_qif(
        manifest,
        &metadata,
        |name| Ok(if name == "norm" { &NORM[..] } else { &WEIGHT }),
        &mut bytes,
    )
    .unwrap();
    bytes
}

#[test]
fn round_trips_manifest_metadata_and_aligned_payloads() {
    let bytes = artifact();
    assert_eq!(&bytes[..8], b"QIF\0\x01\0\0\0");
    let reader = QifReader::open(&bytes).unwrap();
    let (manifest, metadata) = model();
    assert_eq!(reader.manifest().architecture, manifest.architecture);
    assert_eq!(reader.metadata(), &metadata);
    assert_eq!(reader.tensor_bytes("norm").unwrap(), NORM);
    assert_eq!(reader.tensor_bytes("weight").unwrap(), WEIGHT);
    assert!(reader.tensor_bytes("unknown").is_err());
    assert!(reader
        .tensors
        .values()
        .all(|range| range.start % ALIGNMENT == 0));
    assert_eq!(
        reader.manifest().tensors[1].checksum_sha256.as_deref(),
        Some(sha256_hex(&WEIGHT).as_str()),
    );
}

#[test]
fn rejects_unsupported_versions_and_damaged_artifacts() {
    let bytes = artifact();
    let reader = QifReader::open(&bytes).unwrap();
    let damaged_offsets = [
        (4, "unsupported version"),
        (PREFIX, "header checksum mismatch"),
        (reader.tensors["norm"].end, "nonzero tensor padding"),
        (
            reader.tensors["weight"].start,
            "tensor weight checksum mismatch",
        ),
    ];
    for (offset, message) in damaged_offsets {
        let mut damaged = bytes.clone();
        damaged[offset] ^= 1;
        let error = QifReader::open(&damaged)
            .err()
            .expect("damage was accepted");
        assert!(error.to_string().contains(message), "{error}");
    }
    assert!(QifReader::open(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(QifReader::open(&trailing).is_err());
}

#[test]
fn validates_all_descriptors_and_lengths_before_writing() {
    let (mut manifest, metadata) = model();
    let mut bytes = Vec::new();
    assert!(write_qif(manifest.clone(), &metadata, |_| Ok(&NORM), &mut bytes).is_err());
    assert!(bytes.is_empty());
    manifest.tensors[1].descriptor.logical_shape = Shape::from(vec![usize::MAX, 32]);
    assert!(write_qif(manifest, &metadata, |_| Ok(&NORM), &mut bytes).is_err());
    assert!(bytes.is_empty());
}

#[test]
fn imports_legacy_q4_0_without_changing_decoded_values() {
    let block = Q4_0Block {
        scale: 0.5,
        values: std::array::from_fn(|index| ((index * 7 + index / 4) % 16) as u8),
    };
    let tensor = QuantizedTensor4::from_blocks(vec![1, 32], vec![block.clone()]).unwrap();
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("qwen3".to_owned()),
    );
    writer.insert_metadata("qwen3.tie_word_embeddings", GgufValue::Bool(true));
    writer.add_q4_0_tensor("weight", &tensor);
    writer.metadata.remove(QUANT_BLOCK_LAYOUT);
    for (index, pair) in block.values.as_chunks::<2>().0.iter().enumerate() {
        writer.tensors[0].data[index + 2] = pair[0] | (pair[1] << 4);
    }
    let gguf = GgufReader::open(writer.finish().unwrap()).unwrap();
    assert_eq!(gguf.tensor_q4_0("weight").unwrap(), tensor);
    let mut bytes = Vec::new();
    import_gguf(&gguf, "legacy", &mut bytes).unwrap();
    let qif = PackedTensorReader::open_artifact(bytes).unwrap();
    assert_ne!(
        gguf.tensor_bytes("weight").unwrap(),
        qif.tensor_bytes("weight").unwrap()
    );
    assert_eq!(qif.tensor_q4_0("weight").unwrap(), tensor);
}
