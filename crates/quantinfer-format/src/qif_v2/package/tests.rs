use super::*;
use crate::TensorManifest;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};

fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("qif-package-{name}-{}", std::process::id()))
}

fn manifest() -> ModelManifest {
    let mut model = ModelManifest::new("validation", "fixture");
    model.tensors.push(TensorManifest {
        name: "weight".into(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });
    model
}

fn package(name: &str) -> PathBuf {
    let path = path(name);
    let _ = fs::remove_dir_all(&path);
    write_qif_v2_package(
        manifest(),
        &BTreeMap::new(),
        |_| Ok(&[0; 8]),
        &path,
        Some(1024),
    )
    .unwrap();
    path
}

fn edit(root: &Path, change: impl FnOnce(&mut PackageManifest, &mut Vec<u8>)) {
    let mut package: PackageManifest =
        serde_json::from_slice(&fs::read(root.join("qif.json")).unwrap()).unwrap();
    let metadata = fs::read(root.join(&package.metadata_path)).unwrap();
    let mut index = fs::read(root.join(&package.index_path)).unwrap();
    change(&mut package, &mut index);
    package.index_sha256 = crate::sha256_hex(&index);
    package.root_hash.clear();
    package.root_hash = package_root_hash(
        &serde_json::to_vec(&package).unwrap(),
        &metadata,
        &index,
        &package.shards,
    );
    fs::write(root.join(&package.index_path), index).unwrap();
    fs::write(root.join("qif.json"), serde_json::to_vec(&package).unwrap()).unwrap();
}

#[test]
fn malformed_index_lengths_return_errors_without_panicking() {
    let root = package("overflow-regression");
    edit(&root, |_, index| {
        *index = vec![0; 292];
        index[..4].copy_from_slice(b"QIDX");
        for (offset, value) in [(4, 1u32), (8, 1), (12, 1)] {
            index[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (offset, value) in [(16, u64::MAX - 148), (24, 201), (32, 120), (40, 72)] {
            index[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        index[48..52].copy_from_slice(&1000u32.to_le_bytes());
        for (offset, value) in [(236, 8u64), (252, 8)] {
            index[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
    });
    assert!(PackageReader::open(&root).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn manifest_checksum_must_match_the_global_index() {
    let root = package("checksum-regression");
    for checksum in [Some("00".repeat(32)), None] {
        edit(&root, |manifest, _| {
            manifest.model.tensors[0].checksum_sha256 = checksum
        });
        assert!(PackageReader::open(&root)
            .err()
            .unwrap()
            .to_string()
            .contains("checksum disagrees"));
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_rejects_unreferenced_chunks() {
    let root = package("orphan-regression");
    edit(&root, |_, index| {
        let length = index.len();
        index.extend_from_within(length - 72..);
        index[12..16].copy_from_slice(&2u32.to_le_bytes());
        index[40..48].copy_from_slice(&144u64.to_le_bytes());
    });
    assert!(PackageReader::open(&root)
        .err()
        .unwrap()
        .to_string()
        .contains("unreferenced"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_rejects_empty_physical_chunks() {
    let root = package("empty-chunk");
    edit(&root, |_, index| {
        let length = index.len();
        let record = 48 + u64_at(index, 16) as usize + u64_at(index, 24) as usize;
        let mut chunk = index[length - 72..].to_vec();
        for (offset, value) in [(8, 8u64), (16, 0), (24, 0), (32, 0)] {
            chunk[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        chunk[40..72].copy_from_slice(&Sha256::digest([]));
        index.extend_from_slice(&chunk);
        index[12..16].copy_from_slice(&2u32.to_le_bytes());
        index[40..48].copy_from_slice(&144u64.to_le_bytes());
        index[record + 44..record + 48].copy_from_slice(&2u32.to_le_bytes());
    });
    assert!(PackageReader::open(&root)
        .err()
        .unwrap()
        .to_string()
        .contains("empty physical"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn package_alias_shape_must_match_the_physical_tensor() {
    let root = path("alias-shape-regression");
    let _ = fs::remove_dir_all(&root);
    let mut model = manifest();
    model.tensors[0].descriptor.logical_shape = Shape::from(vec![2, 2]);
    let mut alias = model.tensors[0].clone();
    alias.name = "alias".into();
    model.tensors.push(alias);
    write_qif_v2_package(model, &BTreeMap::new(), |_| Ok(&[0; 16]), &root, Some(1024)).unwrap();
    edit(&root, |manifest, index| {
        manifest.model.tensors[1].descriptor.logical_shape = Shape::from(vec![1, 4]);
        let shape_start = 48 + u64_at(index, 16) as usize;
        let alias_record = shape_start + u64_at(index, 24) as usize + 120;
        let entry = shape_start + u32_at(index, alias_record + 12) as usize;
        index[entry + 4..entry + 12].copy_from_slice(&1u64.to_le_bytes());
        index[entry + 12..entry + 20].copy_from_slice(&4u64.to_le_bytes());
    });
    assert!(PackageReader::open(&root)
        .err()
        .unwrap()
        .to_string()
        .contains("alias metadata"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn shard_limit_includes_alignment_and_compression_overhead() {
    for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
        let root = path(&format!("shard-limit-{}", codec.id()));
        let _ = fs::remove_dir_all(&root);
        let mut model = manifest();
        model.tensors.push(TensorManifest {
            name: "bias".into(),
            ..model.tensors[0].clone()
        });
        let a = [0; 8];
        let b = [0, 0, 128, 63, 0, 0, 0, 64];
        write_qif_v2_package_with_codec(
            model,
            &BTreeMap::new(),
            |name| Ok(if name == "weight" { &a } else { &b }),
            &root,
            Some(32),
            codec,
        )
        .unwrap();
        let mut reader = PackageReader::open(&root).unwrap();
        assert!(reader.entries.values().all(|shard| shard.size <= 32));
        assert_eq!(reader.read_range("weight", 0..8).unwrap(), a);
        assert_eq!(reader.read_range("bias", 0..8).unwrap(), b);
        reader.verify().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    let root = path("impossible-codec-limit");
    let _ = fs::remove_dir_all(&root);
    assert!(write_qif_v2_package_with_codec(
        manifest(),
        &BTreeMap::new(),
        |_| Ok(&[0; 8]),
        &root,
        Some(1),
        Codec::Zstd
    )
    .is_err());
    assert!(!root.exists());
}

#[test]
fn package_range_does_not_read_the_unrequested_chunk_tail() {
    let root = package("small-range");
    let mut reader = PackageReader::open(&root).unwrap();
    let shard = root.join(reader.tensor_shard("weight").unwrap());
    fs::OpenOptions::new()
        .write(true)
        .open(shard)
        .unwrap()
        .set_len(4)
        .unwrap();
    assert_eq!(reader.read_range("weight", 0..4).unwrap(), [0; 4]);
    assert!(reader.verify().is_err());
    fs::remove_dir_all(root).unwrap();
}
