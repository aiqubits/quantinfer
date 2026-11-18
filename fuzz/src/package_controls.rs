//! Bounded mutations of PACKAGE JSON, metadata, paths and shard contents.
//! The mirror below reproduces canonical field order solely to repair control
//! hashes after mutations. It is not a second production format implementation.
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{
    qif::{write_qif_package_with_codec, Codec, PackageReader},
    sha256_hex, ModelManifest, PackedTensorReader, TensorManifest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
struct Package {
    schema_version: u32,
    profile: String,
    model: ModelManifest,
    metadata_path: String,
    metadata_sha256: String,
    index_path: String,
    index_sha256: String,
    shards: Vec<Shard>,
    root_hash: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct Shard {
    path: String,
    size: u64,
    sha256: String,
    mesh_axis: u32,
    rank: u32,
    coordinate: Vec<u32>,
    tensors: Vec<String>,
}

struct Case {
    root: PathBuf,
    package: Package,
    json: Vec<u8>,
    metadata: Vec<u8>,
    index: Vec<u8>,
    shard: Vec<u8>,
}

pub struct PackageCases(Vec<Case>);

impl Default for PackageCases {
    fn default() -> Self {
        Self::new()
    }
}

impl PackageCases {
    pub fn new() -> Self {
        Self(
            [Codec::None, Codec::Zstd, Codec::Lz4]
                .into_iter()
                .enumerate()
                .map(|(id, codec)| {
                    let root = super::workspace().join(format!("controls-{id}"));
                    let mut model = ModelManifest::new("fuzz-controls", "fixture");
                    for name in ["weight", "alias"] {
                        model.tensors.push(TensorManifest {
                            name: name.into(),
                            descriptor: TensorStorageDescriptor::floating(
                                DType::F32,
                                Shape::from(vec![2]),
                            )
                            .unwrap(),
                            checksum_sha256: None,
                        });
                    }
                    write_qif_package_with_codec(
                        model,
                        &BTreeMap::new(),
                        |_| Ok(&[0, 0, 128, 63, 0, 0, 0, 64]),
                        &root,
                        Some(4096),
                        codec,
                    )
                    .unwrap();
                    let json = fs::read(root.join("qif.json")).unwrap();
                    let package: Package = serde_json::from_slice(&json).unwrap();
                    assert_eq!(serde_json::to_vec(&package).unwrap(), json);
                    // Fixed outside targets exercise canonicalization and root escape
                    // handling. No path supplied by the fuzzer is ever used for writes.
                    #[cfg(unix)]
                    std::os::unix::fs::symlink(root.parent().unwrap(), root.join("escape"))
                        .unwrap();
                    Case {
                        metadata: fs::read(root.join("metadata.qmb")).unwrap(),
                        index: fs::read(root.join("index.qifi")).unwrap(),
                        shard: fs::read(root.join("shard-00000.qifd")).unwrap(),
                        root,
                        package,
                        json,
                    }
                })
                .collect(),
        )
    }

    pub fn exercise(&self, data: &[u8]) {
        let case = &self.0[data[0] as usize % self.0.len()];
        let payload = &data[2..];
        let mut package = case.package.clone();
        let mut metadata = case.metadata.as_slice();
        let mut shard = case.shard.as_slice();
        let raw_json = match data[1] % 5 {
            0 => Some(payload),
            1 => {
                // serde's own depth bound applies before constructing the typed
                // mirror; invalid/duplicate/noncanonical raw JSON uses mode 0.
                let Ok(value) = serde_json::from_slice::<Package>(payload) else {
                    return;
                };
                package = value;
                None
            }
            2 => {
                metadata = payload;
                None
            }
            3 => {
                shard = payload;
                package.shards[0].size = shard.len() as u64;
                package.shards[0].sha256 = sha256_hex(shard);
                None
            }
            _ => {
                let Some((&field, path)) = payload.split_first() else {
                    return;
                };
                let Ok(path) = std::str::from_utf8(path) else {
                    return;
                };
                match field % 3 {
                    0 => package.metadata_path = path.into(),
                    1 => package.index_path = path.into(),
                    _ => package.shards[0].path = path.into(),
                }
                None
            }
        };
        package.metadata_sha256 = sha256_hex(metadata);
        package.index_sha256 = sha256_hex(&case.index);
        package.root_hash.clear();
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(&package).unwrap());
        hash.update(metadata);
        hash.update(&case.index);
        for shard in &package.shards {
            hash.update(shard.sha256.as_bytes());
        }
        package.root_hash = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let repaired = serde_json::to_vec(&package).unwrap();
        fs::write(case.root.join("metadata.qmb"), metadata).unwrap();
        fs::write(case.root.join("shard-00000.qifd"), shard).unwrap();
        fs::write(case.root.join("qif.json"), raw_json.unwrap_or(&repaired)).unwrap();
        let native = PackageReader::open(&case.root);
        let packed = PackedTensorReader::open_path(&case.root);
        if let Ok(native) = &native {
            if packed.is_err() {
                assert!(
                    matches!(native.metadata().get("quantinfer.qif.ggml_dimension_order"), Some(value) if !matches!(value, quantinfer_format::qif::MetadataValue::Bool(_)))
                );
            }
        } else {
            assert!(packed.is_err());
        }
        if let (Ok(mut native), Ok(packed)) = (native, packed) {
            // Never allocate or verify unbounded mutated declarations.
            if !super::bounded_lengths(native.tensors().iter().map(|tensor| tensor.raw_length)) {
                return;
            }
            assert_eq!(native.verify().is_ok(), packed.verify_qif().is_ok());
            for tensor in native.tensors().to_vec().iter().take(8) {
                let end = tensor.raw_length.min(4096) as usize;
                let _ = native.read_range(&tensor.name, 0..end);
                let _ = packed.tensor_data(&tensor.name);
            }
            assert_eq!(packed.cached_tensor_bytes(), 0);
        }
    }

    pub fn write_seeds(&self, output: &Path) {
        fs::create_dir_all(output).unwrap();
        for (codec, case) in self.0.iter().enumerate() {
            for (mode, bytes) in [
                (0, &case.json),
                (1, &case.json),
                (2, &case.metadata),
                (3, &case.shard),
            ] {
                fs::write(
                    output.join(format!("{codec}-{mode}")),
                    [&[codec as u8, mode], bytes.as_slice()].concat(),
                )
                .unwrap();
            }
            let duplicate = String::from_utf8(case.json.clone()).unwrap().replacen(
                "\"schema_version\":1",
                "\"schema_version\":1,\"schema_version\":1",
                1,
            );
            fs::write(
                output.join(format!("{codec}-duplicate-json")),
                [&[codec as u8, 0], duplicate.as_bytes()].concat(),
            )
            .unwrap();
            let deep = format!("{}0{}", "[".repeat(40), "]".repeat(40));
            fs::write(
                output.join(format!("{codec}-deep-json")),
                [&[codec as u8, 0], deep.as_bytes()].concat(),
            )
            .unwrap();
            for field in 0..3 {
                for (id, path) in [
                    "../qif.json",
                    "/etc/passwd",
                    "escape/qif.json",
                    "metadata.qmb",
                    "index.qifi",
                    "shard-00000.qifd",
                    "missing",
                    ".",
                ]
                .iter()
                .enumerate()
                {
                    fs::write(
                        output.join(format!("{codec}-path-{field}-{id}")),
                        [&[codec as u8, 4, field], path.as_bytes()].concat(),
                    )
                    .unwrap();
                }
            }
        }
    }
}
