use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif_v2::{write_qif_v2, QifV2MetadataValue, QifV2Reader};
use quantinfer_format::{ModelManifest, TensorManifest};

pub(crate) fn run() {
    let payload = (0..(1024 * 1024 / 4))
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let mut manifest = ModelManifest::new("qif-v2-benchmark", "fixture");
    manifest.tensors.push(TensorManifest {
        name: "weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(
            DType::F32,
            Shape::from(vec![payload.len() / 4]),
        )
        .unwrap(),
        checksum_sha256: None,
    });
    let metadata = BTreeMap::<String, QifV2MetadataValue>::new();
    let mut artifact = Vec::new();
    write_qif_v2(
        manifest,
        &metadata,
        |_| Ok(payload.as_slice()),
        &mut artifact,
    )
    .expect("benchmark artifact should be valid");
    let started = Instant::now();
    let reader = QifV2Reader::open(&artifact).expect("benchmark reader should open");
    println!("qif2_open_ns={}", started.elapsed().as_nanos());
    let started = Instant::now();
    black_box(
        reader
            .read_range("weight", 0..4096)
            .expect("range should read"),
    );
    println!("qif2_first_range_ns={}", started.elapsed().as_nanos());
    let started = Instant::now();
    reader.verify().expect("benchmark verification should pass");
    println!("qif2_full_verify_ns={}", started.elapsed().as_nanos());
    println!("qif2_artifact_bytes={}", artifact.len());
}
