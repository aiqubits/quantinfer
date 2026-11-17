#![no_main]

use libfuzzer_sys::fuzz_target;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif::{write_qif_package, PackageReader};
use quantinfer_format::{ModelManifest, TensorManifest};
use quantinfer_format_fuzz::{workspace, MAX_INPUT};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::PathBuf, sync::OnceLock};

struct PackageCase {
    root: PathBuf,
    unsigned: String,
    original_index_hash: String,
    metadata: Vec<u8>,
    shard_hash: String,
}

impl PackageCase {
    fn new() -> Self {
        let root = workspace().join("package");
        let mut model = ModelManifest::new("fuzz", "fixture");
        model.tensors.push(TensorManifest {
            name: "weight".into(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))
                .unwrap(),
            checksum_sha256: None,
        });
        write_qif_package(
            model,
            &BTreeMap::new(),
            |_| Ok(&[0, 0, 128, 63, 0, 0, 0, 64]),
            &root,
            Some(1024),
        )
        .unwrap();
        let text = fs::read_to_string(root.join("qif.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        let root_hash = value["root_hash"].as_str().unwrap();
        Self {
            metadata: fs::read(root.join("metadata.qmb")).unwrap(),
            unsigned: text.replace(
                &format!("\"root_hash\":\"{root_hash}\""),
                "\"root_hash\":\"\"",
            ),
            original_index_hash: value["index_sha256"].as_str().unwrap().into(),
            shard_hash: value["shards"][0]["sha256"].as_str().unwrap().into(),
            root,
        }
    }

    fn exercise(&self, index: &[u8]) {
        let new_hash = quantinfer_format::sha256_hex(index);
        // Preserve the typed package serializer's field order, rather than
        // reserializing through a JSON map with a different canonical order.
        let unsigned = self.unsigned.replace(
            &format!("\"index_sha256\":\"{}\"", self.original_index_hash),
            &format!("\"index_sha256\":\"{new_hash}\""),
        );
        let mut hash = Sha256::new();
        hash.update(unsigned.as_bytes());
        hash.update(&self.metadata);
        hash.update(index);
        hash.update(self.shard_hash.as_bytes());
        let root_hash: String = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let text = unsigned.replace(
            "\"root_hash\":\"\"",
            &format!("\"root_hash\":\"{root_hash}\""),
        );
        fs::write(self.root.join("index.qifi"), index).unwrap();
        fs::write(self.root.join("qif.json"), text).unwrap();
        if let Ok(mut reader) = PackageReader::open(&self.root) {
            // The immutable manifest fixes this target's output at 8 bytes.
            let _ = reader.read_range("weight", 0..8);
            let _ = reader.verify();
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT {
        return;
    }
    static CASE: OnceLock<PackageCase> = OnceLock::new();
    CASE.get_or_init(PackageCase::new).exercise(data);
});
