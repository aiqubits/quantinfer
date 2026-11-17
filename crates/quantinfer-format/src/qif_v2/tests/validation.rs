use std::collections::BTreeMap;
use std::fs;

use quantinfer_core::{
    DType, QuantizationAxis, QuantizationScheme, ScaleType, Shape, TensorStorageDescriptor,
    ZeroPointType,
};
use sha2::{Digest, Sha256};

use super::super::{
    codec, metadata, write_qif_v2, write_qif_v2_with_codec, Codec, FileReader, MetadataValue,
    Reader,
};
use super::sample_manifest;

fn u64_at(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}

fn refresh_root(bytes: &mut [u8]) {
    let mut hash = Sha256::new();
    for offset in [24, 40, 56] {
        let start = u64_at(bytes, offset);
        let length = u64_at(bytes, offset + 8);
        hash.update(&bytes[start..start + length]);
    }
    bytes[88..120].copy_from_slice(&hash.finalize());
}

fn single(payload: &[u8]) -> Vec<u8> {
    let mut manifest = sample_manifest();
    manifest.tensors[0].descriptor =
        TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![payload.len() / 4]))
            .unwrap();
    let mut bytes = Vec::new();
    write_qif_v2(manifest, &BTreeMap::new(), |_| Ok(payload), &mut bytes).unwrap();
    bytes
}

#[test]
fn rejects_unreferenced_and_overlapping_chunks_with_consistent_root() {
    let mut bytes = single(&[0; 8]);
    let index = u64_at(&bytes, 56);
    let length = u64_at(&bytes, 64);
    let chunk = bytes[index + length - 72..index + length].to_vec();
    bytes[index + length..index + length + 72].copy_from_slice(&chunk);
    bytes[index + length + 40..index + length + 72].fill(0);
    bytes[index + 12..index + 16].copy_from_slice(&2u32.to_le_bytes());
    bytes[index + 40..index + 48].copy_from_slice(&144u64.to_le_bytes());
    bytes[64..72].copy_from_slice(&((length + 72) as u64).to_le_bytes());
    refresh_root(&mut bytes);
    assert!(Reader::open(&bytes)
        .err()
        .unwrap()
        .to_string()
        .contains("unreferenced"));
    let path = std::env::temp_dir().join(format!("qif-orphan-{}.qif", std::process::id()));
    fs::write(&path, bytes).unwrap();
    assert!(FileReader::open(&path)
        .err()
        .unwrap()
        .to_string()
        .contains("unreferenced"));
    fs::remove_file(path).unwrap();
}

#[test]
fn rejects_noncanonical_json_after_root_validation() {
    let mut bytes = single(&[0; 8]);
    let start = u64_at(&bytes, 24);
    let length = u64_at(&bytes, 32);
    // Valid JSON with an extra leading space; unlike corrupt JSON this reaches
    // the canonicalization check after a correct root hash and typed parse.
    bytes.copy_within(start..start + length, start + 1);
    bytes[start] = b' ';
    bytes[32..40].copy_from_slice(&((length + 1) as u64).to_le_bytes());
    refresh_root(&mut bytes);
    assert!(Reader::open(&bytes)
        .err()
        .unwrap()
        .to_string()
        .contains("canonical JSON"));
}

#[test]
fn rejects_each_incompatible_quantized_encoding_field_on_write_and_read() {
    let descriptor = TensorStorageDescriptor::quantized(
        DType::Q4_0,
        QuantizationScheme::Q4_0,
        Shape::from(vec![32]),
        32,
    )
    .unwrap();
    for field in 0..7 {
        let mut manifest = sample_manifest();
        manifest.quantization = QuantizationScheme::Q4_0;
        manifest.tensors[0].descriptor = descriptor.clone();
        let mut valid = Vec::new();
        write_qif_v2(
            manifest.clone(),
            &BTreeMap::new(),
            |_| Ok(&[0; 18]),
            &mut valid,
        )
        .unwrap();
        let d = &mut manifest.tensors[0].descriptor;
        match field {
            0 => d.block_size = Some(16),
            1 => d.scale_type = Some(ScaleType::F32),
            2 => d.zero_point_type = ZeroPointType::U8,
            3 => d.quantization_axis = QuantizationAxis::Axis(0),
            4 => d.group_size = Some(16),
            5 => d.packing = "unknown".into(),
            _ => d.quantization_axis = QuantizationAxis::Axis(999),
        }
        assert!(
            write_qif_v2(
                manifest.clone(),
                &BTreeMap::new(),
                |_| Ok(&[0; 18]),
                &mut Vec::new()
            )
            .is_err(),
            "writer accepted field {field}"
        );
        // Mutate the valid artifact independently of writer validation, retaining
        // its correct tensor hash and recomputing the root.
        let original = Reader::open(&valid).unwrap().manifest().clone();
        manifest.tensors[0].checksum_sha256 = original.tensors[0].checksum_sha256.clone();
        let encoded = serde_json::to_vec(&manifest).unwrap();
        let start = u64_at(&valid, 24);
        assert!(start + encoded.len() < u64_at(&valid, 40));
        valid[start..start + encoded.len()].copy_from_slice(&encoded);
        valid[32..40].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        refresh_root(&mut valid);
        assert!(
            Reader::open(&valid).is_err(),
            "reader accepted field {field}"
        );
    }
}

#[test]
fn metadata_depth_boundary_round_trips_and_rejects_one_more_level() {
    let mut value = MetadataValue::U32(1);
    for _ in 0..31 {
        value = MetadataValue::Map(BTreeMap::from([("key".into(), value)]));
    }
    let values = BTreeMap::from([("root".into(), value.clone())]);
    assert_eq!(
        metadata::decode(&metadata::encode(&values).unwrap()).unwrap(),
        values
    );
    let excessive = BTreeMap::from([(
        "root".into(),
        MetadataValue::Map(BTreeMap::from([("key".into(), value)])),
    )]);
    assert!(metadata::encode(&excessive).is_err());
}

#[test]
fn public_readers_bound_compressed_output_to_declared_raw_length() {
    for codec_id in [Codec::Zstd, Codec::Lz4] {
        let mut bytes = Vec::new();
        write_qif_v2_with_codec(
            sample_manifest(),
            &BTreeMap::new(),
            |_| Ok(&[0; 8]),
            &mut bytes,
            codec_id,
        )
        .unwrap();
        let packed = codec::compress(codec_id.id(), &vec![0; 8 * 1024 * 1024]).unwrap();
        let index = u64_at(&bytes, 56);
        let strings = u64_at(&bytes, index + 16);
        let shapes = u64_at(&bytes, index + 24);
        let record = index + 48 + strings + shapes;
        let chunk = record + 120;
        let payload = u64_at(&bytes, 72);
        bytes.resize(payload + packed.len(), 0);
        bytes[payload..].copy_from_slice(&packed);
        bytes[80..88].copy_from_slice(&(packed.len() as u64).to_le_bytes());
        for offset in [record + 56, chunk + 32] {
            bytes[offset..offset + 8].copy_from_slice(&(packed.len() as u64).to_le_bytes());
        }
        bytes[chunk + 40..chunk + 72].copy_from_slice(&Sha256::digest(&packed));
        refresh_root(&mut bytes);
        let error = Reader::open(&bytes)
            .unwrap()
            .tensor_data("weight")
            .unwrap_err()
            .to_string();
        if codec_id == Codec::Zstd {
            assert!(error.contains("9 bytes, expected 8"), "{error}");
        } else {
            assert!(error.contains("size prefix"), "{error}");
        }
        let path = std::env::temp_dir().join(format!(
            "qif-bomb-{}-{}.qif",
            codec_id.id(),
            std::process::id()
        ));
        fs::write(&path, &bytes).unwrap();
        assert!(FileReader::open(&path)
            .unwrap()
            .read_range("weight", 0..8)
            .is_err());
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn file_range_does_not_read_the_unrequested_chunk_tail() {
    let bytes = single(&vec![0; 4 * 1024 * 1024]);
    let path = std::env::temp_dir().join(format!("qif-small-range-{}.qif", std::process::id()));
    fs::write(&path, &bytes).unwrap();
    let mut reader = FileReader::open(&path).unwrap();
    // Inject a missing tail after control validation. Reading a prefix must not
    // attempt to read the missing tail; full verification must still reject it.
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len((u64_at(&bytes, 72) + 8) as u64)
        .unwrap();
    assert_eq!(reader.read_range("weight", 0..8).unwrap(), [0; 8]);
    assert!(reader.verify().is_err());
    fs::remove_file(path).unwrap();
}

#[test]
fn packed_reader_preserves_native_maps_and_checks_structural_metadata_types() {
    let map = MetadataValue::Map(BTreeMap::from([(
        "prefetch".into(),
        MetadataValue::Bool(true),
    )]));
    let values = BTreeMap::from([("optional.hints".into(), map.clone())]);
    let mut bytes = Vec::new();
    write_qif_v2(sample_manifest(), &values, |_| Ok(&[0; 8]), &mut bytes).unwrap();
    let reader = crate::GgufReader::open_artifact(bytes).unwrap();
    assert_eq!(reader.qif_metadata().unwrap(), &values);
    assert_eq!(reader.tensor_bytes("weight").unwrap(), [0; 8]);
    let values = BTreeMap::from([("quantinfer.qif.ggml_dimension_order".into(), map)]);
    let mut bytes = Vec::new();
    write_qif_v2(sample_manifest(), &values, |_| Ok(&[0; 8]), &mut bytes).unwrap();
    assert!(crate::GgufReader::open_artifact(bytes)
        .unwrap_err()
        .to_string()
        .contains("must be boolean"));
}

#[test]
fn all_core_encodings_round_trip_with_codecs_aliases_and_cross_shard_ranges() {
    use crate::GgmlType;
    let encodings = [
        (DType::F32, GgmlType::F32, QuantizationScheme::None, 0),
        (DType::F16, GgmlType::F16, QuantizationScheme::None, 0),
        (DType::BF16, GgmlType::BF16, QuantizationScheme::None, 0),
        (DType::Q8_0, GgmlType::Q8_0, QuantizationScheme::Q8_0, 32),
        (DType::Q4_0, GgmlType::Q4_0, QuantizationScheme::Q4_0, 32),
        (DType::Q4_K, GgmlType::Q4_K, QuantizationScheme::Q4_K, 256),
        (DType::Q5_K, GgmlType::Q5_K, QuantizationScheme::Q5_K, 256),
        (DType::Q6_K, GgmlType::Q6_K, QuantizationScheme::Q6_K, 256),
    ];
    for (dtype, physical, scheme, block_size) in encodings {
        for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
            let mut manifest = sample_manifest();
            manifest.quantization = scheme;
            let shape = Shape::from(vec![256]);
            manifest.tensors[0].descriptor = if block_size == 0 {
                TensorStorageDescriptor::floating(dtype, shape).unwrap()
            } else {
                TensorStorageDescriptor::quantized(dtype, scheme, shape, block_size).unwrap()
            };
            manifest.tensors.push(crate::TensorManifest {
                name: "alias".into(),
                ..manifest.tensors[0].clone()
            });
            let payload = vec![0; crate::tensor_byte_size(physical, &[256]).unwrap()];
            let mut single = Vec::new();
            write_qif_v2_with_codec(
                manifest.clone(),
                &BTreeMap::new(),
                |_| Ok(&payload),
                &mut single,
                codec,
            )
            .unwrap();
            let reader = Reader::open(&single).unwrap();
            reader.verify().unwrap();
            assert_eq!(reader.tensor_info("alias").unwrap().alias_of, Some(0));
            let root = std::env::temp_dir().join(format!(
                "qif-matrix-{dtype:?}-{}-{}",
                codec.id(),
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&root);
            super::super::write_qif_v2_package_with_codec(
                manifest,
                &BTreeMap::new(),
                |_| Ok(&payload),
                &root,
                Some(64),
                codec,
            )
            .unwrap();
            let mut package = super::super::PackageReader::open(&root).unwrap();
            package.verify().unwrap();
            assert_eq!(package.tensor_info("alias").unwrap().alias_of, Some(0));
            let end = payload.len().min(100);
            for name in ["weight", "alias"] {
                assert_eq!(
                    package.read_range(name, 1..end).unwrap(),
                    reader.read_range(name, 1..end).unwrap()
                );
            }
            for entry in fs::read_dir(&root).unwrap() {
                let entry = entry.unwrap();
                if entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "qifd")
                {
                    assert!(entry.metadata().unwrap().len() <= 64);
                }
            }
            fs::remove_dir_all(root).unwrap();
        }
    }
}

#[test]
fn ggml_quantized_projection_preserves_axis_layout_bytes_and_shape() {
    use crate::{GgufReader, GgufValue, GgufWriter};
    use quantinfer_core::{Q4_0Block, QuantizedTensor4};
    let quantized = QuantizedTensor4::from_blocks(
        vec![32, 2],
        vec![
            Q4_0Block {
                scale: 0.5,
                values: [8; 32],
            },
            Q4_0Block {
                scale: 0.25,
                values: [4; 32],
            },
        ],
    )
    .unwrap();
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("llama".into()));
    writer.insert_metadata("general.file_type", GgufValue::U32(2));
    writer.insert_metadata("llama.embedding_length", GgufValue::U32(32));
    writer.add_q4_0_tensor("blk.0.attn_q.weight", &quantized);
    writer
        .add_f32_tensor("token_embd.weight", vec![32, 4], &[0.0; 128])
        .unwrap();
    let gguf = GgufReader::open(writer.finish().unwrap()).unwrap();
    let mut output = Vec::new();
    super::super::import_gguf(&gguf, "projection", &mut output).unwrap();
    let reader = Reader::open(&output).unwrap();
    reader.verify().unwrap();
    let descriptor = &reader.manifest().tensors[0].descriptor;
    assert_eq!(descriptor.physical_layout, crate::GGML_OUTPUT_ROWS_LAYOUT);
    assert_eq!(descriptor.quantization_axis, QuantizationAxis::Axis(0));
    let root = std::env::temp_dir().join(format!("qif-ggml-projection-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    super::super::write_qif_v2_package(
        reader.manifest().clone(),
        reader.metadata(),
        |name| reader.tensor_bytes(name),
        &root,
        Some(64),
    )
    .unwrap();
    let mut package = super::super::PackageReader::open(&root).unwrap();
    package.verify().unwrap();
    assert_eq!(
        package.read_range("blk.0.attn_q.weight", 0..36).unwrap(),
        reader.read_range("blk.0.attn_q.weight", 0..36).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
    let packed = GgufReader::open_artifact(output).unwrap();
    assert_eq!(
        packed.tensor_info("blk.0.attn_q.weight").unwrap(),
        gguf.tensor_info("blk.0.attn_q.weight").unwrap()
    );
    assert_eq!(
        packed.tensor_q4_0("blk.0.attn_q.weight").unwrap(),
        quantized
    );
}
