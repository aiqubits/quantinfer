use super::super::{codec, write_qif_v2_with_codec, Codec, FileReader, Reader};
use super::{
    sample_manifest,
    validation::{refresh_root, u64_at},
};
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs};

#[test]
fn compressed_chunks_bound_range_decoding_and_aliases_share_materialized_storage() {
    let payload = vec![0; codec::DEFAULT_CHUNK_BYTES + 128];
    let mut model = sample_manifest();
    model.tensors[0].descriptor =
        TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![payload.len() / 4]))
            .unwrap();
    model.tensors.push(crate::TensorManifest {
        name: "alias".into(),
        ..model.tensors[0].clone()
    });
    for codec in [Codec::Zstd, Codec::Lz4] {
        let mut bytes = Vec::new();
        write_qif_v2_with_codec(
            model.clone(),
            &BTreeMap::new(),
            |_| Ok(&payload),
            &mut bytes,
            codec,
        )
        .unwrap();
        let reader = Reader::open(&bytes).unwrap();
        assert_eq!(reader.tensor_info("weight").unwrap().chunk_count, 2);
        let start = super::super::codec::DEFAULT_CHUNK_BYTES - 4;
        assert_eq!(
            reader.read_range("alias", start..start + 8).unwrap(),
            [0; 8]
        );
        reader.verify().unwrap();
        // The compressed file fits this budget, but its decoded weights do not.
        let too_small = bytes.len() + payload.len() - 1;
        assert!(crate::GgufReader::open_artifact_with_memory_budget(
            bytes.clone(),
            Some(too_small)
        )
        .unwrap_err()
        .to_string()
        .contains("memory budget"));
        // Aliases share storage and therefore do not double the estimate.
        let budget = bytes.len() + payload.len();
        let packed =
            crate::GgufReader::open_artifact_with_memory_budget(bytes, Some(budget)).unwrap();
        assert_eq!(
            packed.tensor_info("alias").unwrap().offset,
            packed.tensor_info("weight").unwrap().offset
        );
        assert_eq!(
            packed.tensor_bytes("alias").unwrap().as_ptr(),
            packed.tensor_bytes("weight").unwrap().as_ptr()
        );
        assert_eq!(packed.tensor_bytes("weight").unwrap(), payload);
    }
}

fn small_compressed() -> Vec<u8> {
    let mut bytes = Vec::new();
    write_qif_v2_with_codec(
        sample_manifest(),
        &BTreeMap::new(),
        |_| Ok(&[0; 8]),
        &mut bytes,
        Codec::Lz4,
    )
    .unwrap();
    bytes
}

#[test]
fn declared_decoded_length_cannot_trigger_an_unbounded_allocation() {
    let mut bytes = small_compressed();
    let length = codec::MAX_DECODED_CHUNK_BYTES + 4;
    let mut model = Reader::open(&bytes).unwrap().manifest().clone();
    model.tensors[0].descriptor.logical_shape = Shape::from(vec![length / 4]);
    let manifest = serde_json::to_vec(&model).unwrap();
    let manifest_start = u64_at(&bytes, 24);
    bytes[manifest_start..manifest_start + manifest.len()].copy_from_slice(&manifest);
    bytes[32..40].copy_from_slice(&(manifest.len() as u64).to_le_bytes());
    let index = u64_at(&bytes, 56);
    let shape = index + 48 + u64_at(&bytes, index + 16);
    let record = shape + u64_at(&bytes, index + 24);
    let chunk = record + 120;
    bytes[shape + 4..shape + 12].copy_from_slice(&((length / 4) as u64).to_le_bytes());
    for offset in [record + 48, chunk + 16] {
        bytes[offset..offset + 8].copy_from_slice(&(length as u64).to_le_bytes());
    }
    let payload = u64_at(&bytes, 72);
    bytes[payload..payload + 4].copy_from_slice(&(length as u32).to_le_bytes());
    let stored = u64_at(&bytes, chunk + 32);
    let digest = Sha256::digest(&bytes[payload..payload + stored]);
    bytes[chunk + 40..chunk + 72].copy_from_slice(&digest);
    refresh_root(&mut bytes);
    assert!(Reader::open(&bytes)
        .err()
        .unwrap()
        .to_string()
        .contains("resource limits"));
    let path = std::env::temp_dir().join(format!("qif-decode-limit-{}.qif", std::process::id()));
    fs::write(&path, bytes).unwrap();
    assert!(FileReader::open(&path)
        .err()
        .unwrap()
        .to_string()
        .contains("resource limits"));
    fs::remove_file(path).unwrap();
}

#[test]
fn stored_compressed_length_is_checked_before_reading_a_sparse_payload() {
    let mut bytes = small_compressed();
    let stored = codec::MAX_STORED_CHUNK_BYTES + 1;
    let index = u64_at(&bytes, 56);
    let record = index + 48 + u64_at(&bytes, index + 16) + u64_at(&bytes, index + 24);
    let chunk = record + 120;
    for offset in [record + 56, chunk + 32, 80] {
        bytes[offset..offset + 8].copy_from_slice(&(stored as u64).to_le_bytes());
    }
    refresh_root(&mut bytes);
    let path = std::env::temp_dir().join(format!("qif-stored-limit-{}.qif", std::process::id()));
    fs::write(&path, &bytes).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len((u64_at(&bytes, 72) + stored) as u64)
        .unwrap();
    assert!(FileReader::open(&path)
        .err()
        .unwrap()
        .to_string()
        .contains("resource limits"));
    fs::remove_file(path).unwrap();
}
