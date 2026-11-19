use super::super::*;
use crate::qif_v2::{write_qif_v2_with_codec, FileReader, MappedFile, Reader};

fn number(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}

fn set_number(bytes: &mut [u8], offset: usize, value: usize) {
    bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
}

struct Fixture {
    model: ModelManifest,
    metadata: Vec<u8>,
    index: Vec<u8>,
    payload: Vec<u8>,
    single: Vec<u8>,
}

fn fixture(codec: Codec, alignment: usize) -> Fixture {
    let raw = [0, 0, 128, 63, 0, 0, 0, 64];
    let mut model = super::manifest();
    let mut alias = model.tensors[0].clone();
    alias.name = "alias".into();
    model.tensors.push(alias);
    let mut original = Vec::new();
    write_qif_v2_with_codec(model, &BTreeMap::new(), |_| Ok(&raw), &mut original, codec).unwrap();
    let native = Reader::open(&original).unwrap();
    let model = native.manifest().clone();
    let section = |field| {
        original[number(&original, field)..number(&original, field) + number(&original, field + 8)]
            .to_vec()
    };
    let manifest = section(24);
    let metadata = section(40);
    let mut index = section(56);
    let records = 48 + number(&index, 16) + number(&index, 24);
    let chunks = records + number(&index, 32);
    index.truncate(chunks);
    index[12..16].copy_from_slice(&2_u32.to_le_bytes());
    set_number(&mut index, 40, 2 * CHUNK_RECORD_SIZE);
    index[records + 44..records + 48].copy_from_slice(&2_u32.to_le_bytes());
    let mut payload = Vec::new();
    let mut total = 0;
    for (i, part) in raw.as_chunks::<4>().0.iter().enumerate() {
        let stored = compress(codec.id(), part).unwrap();
        let offset = payload.len().next_multiple_of(alignment);
        payload.resize(offset, 0);
        payload.extend_from_slice(&stored);
        let mut chunk = [0; CHUNK_RECORD_SIZE];
        chunk[4..8].copy_from_slice(&codec.id().to_le_bytes());
        for (field, value) in [(8, i * 4), (16, 4), (24, offset), (32, stored.len())] {
            set_number(&mut chunk, field, value);
        }
        chunk[40..72].copy_from_slice(&Sha256::digest(&stored));
        index.extend_from_slice(&chunk);
        total += stored.len();
    }
    // Both the physical and alias records describe the same encoded length.
    for record in [records, records + TENSOR_RECORD_SIZE] {
        set_number(&mut index, record + 56, total);
    }
    let mut single = original[..144].to_vec();
    for (field, data) in [
        (24, &manifest),
        (40, &metadata),
        (56, &index),
        (72, &payload),
    ] {
        let offset = single.len().next_multiple_of(alignment);
        single.resize(offset, 0);
        single.extend_from_slice(data);
        set_number(&mut single, field, offset);
        set_number(&mut single, field + 8, data.len());
    }
    let mut root = Sha256::new();
    root.update(&manifest);
    root.update(&metadata);
    root.update(&index);
    single[88..120].copy_from_slice(&root.finalize());
    Fixture {
        model,
        metadata,
        index,
        payload,
        single,
    }
}

fn publish_package(path: &Path, f: &Fixture) {
    fs::create_dir(path).unwrap();
    fs::write(path.join("metadata.qmb"), &f.metadata).unwrap();
    fs::write(path.join("index.qifi"), &f.index).unwrap();
    fs::write(path.join("shard-00000.qifd"), &f.payload).unwrap();
    let mut package = PackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        profile: PACKAGE_PROFILE.into(),
        model: f.model.clone(),
        metadata_path: "metadata.qmb".into(),
        metadata_sha256: crate::sha256_hex(&f.metadata),
        index_path: "index.qifi".into(),
        index_sha256: crate::sha256_hex(&f.index),
        shards: vec![ShardManifest {
            path: "shard-00000.qifd".into(),
            size: f.payload.len() as u64,
            sha256: crate::sha256_hex(&f.payload),
            mesh_axis: 0,
            rank: 1,
            coordinate: vec![0],
            tensors: vec!["alias".into(), "weight".into()],
        }],
        root_hash: String::new(),
    };
    package.root_hash = package_root_hash(
        &serde_json::to_vec(&package).unwrap(),
        &f.metadata,
        &f.index,
        &package.shards,
    );
    fs::write(path.join("qif.json"), serde_json::to_vec(&package).unwrap()).unwrap();
}

#[test]
#[allow(unsafe_code)]
fn large_chunk_alignment_preserves_ranges_aliases_and_all_readers() {
    let raw = [0, 0, 128, 63, 0, 0, 0, 64];
    for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
        for alignment in [4096, 65536, 2 * 1024 * 1024] {
            let f = fixture(codec, alignment);
            let reader = Reader::open(&f.single).unwrap();
            reader.verify().unwrap();
            for name in ["weight", "alias"] {
                assert_eq!(reader.tensor_data(name).unwrap(), raw);
                assert_eq!(reader.read_range(name, 2..6).unwrap(), raw[2..6]);
            }
            let path = super::path(&format!("alignment-{}-{alignment}", codec.id()));
            fs::write(&path, &f.single).unwrap();
            let mut reader = FileReader::open(&path).unwrap();
            reader.verify().unwrap();
            assert_eq!(reader.read_range("alias", 2..6).unwrap(), raw[2..6]);
            {
                // SAFETY: this test owns the path and does not mutate or remove
                // the file until the mapping and every borrowed view are gone.
                let mapping = unsafe { MappedFile::open(&path) }.unwrap();
                mapping.reader().unwrap().verify().unwrap();
                assert_eq!(
                    mapping.reader().unwrap().read_range("alias", 0..8).unwrap(),
                    raw
                );
            }
            let packed = crate::PackedTensorReader::open_path(&path).unwrap();
            packed.verify_qif().unwrap();
            assert_eq!(
                packed.tensor_f32("alias").unwrap().as_f32().unwrap(),
                [1.0, 2.0]
            );
            drop((reader, packed));
            fs::remove_file(&path).unwrap();
            publish_package(&path, &f);
            let mut package = PackageReader::open(&path).unwrap();
            package.verify().unwrap();
            assert_eq!(package.read_range("alias", 2..6).unwrap(), raw[2..6]);
            let packed = crate::PackedTensorReader::open_path(&path).unwrap();
            packed.verify_qif().unwrap();
            assert_eq!(
                packed.tensor_f32("weight").unwrap().as_f32().unwrap(),
                [1.0, 2.0]
            );
            drop((package, packed));
            fs::remove_dir_all(path).unwrap();
        }
    }
}
