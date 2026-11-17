use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{qif::write_qif, GgufWriter, ModelManifest, TensorManifest};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn external_formats_use_separate_workers_and_matching_range_payloads() {
    let root = std::env::temp_dir().join(format!(
        "qif-bench-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let qif = root.join("model.qif");
    let safe = root.join("model.safetensors");
    let gguf = root.join("model.gguf");
    let payload = [0, 0, 128, 63, 0, 0, 0, 64];
    let mut manifest = ModelManifest::new("benchmark", "fixture");
    manifest.tensors.push(TensorManifest {
        name: "weight".into(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap(),
        checksum_sha256: None,
    });
    write_qif(
        manifest,
        &BTreeMap::new(),
        |_| Ok(&payload),
        &mut fs::File::create(&qif).unwrap(),
    )
    .unwrap();
    let mut writer = GgufWriter::new();
    writer
        .add_f32_tensor("weight", vec![2], &[1.0, 2.0])
        .unwrap();
    fs::write(&gguf, writer.finish().unwrap()).unwrap();
    let header = br#"{"source.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&payload);
    fs::write(&safe, bytes).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_quantinfer-xtask"))
        .arg("qif-bench")
        .arg("--qif")
        .arg(&qif)
        .arg("--safetensors")
        .arg(&safe)
        .arg("--gguf")
        .arg(&gguf)
        .args([
            "--tensor",
            "weight",
            "--safetensors-tensor",
            "source.weight",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let metrics = stdout
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(metrics["benchmark_process_isolation"], "1");
    assert_eq!(metrics["benchmark_cache_state"], "uncontrolled");
    let mut workers = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for format in ["qif", "safetensors", "gguf"] {
        workers.insert(metrics[format!("{format}_process_id").as_str()]);
        hashes.insert(metrics[format!("{format}_range_sha256").as_str()]);
        assert_eq!(metrics[format!("{format}_range_bytes").as_str()], "8");
        assert_eq!(metrics[format!("{format}_dtype").as_str()], "F32");
        let ns = |metric: &str| {
            metrics[format!("{format}_{metric}_ns").as_str()]
                .parse::<u128>()
                .unwrap()
        };
        assert!(ns("open_to_first_range") >= ns("open") + ns("first_range"));
    }
    assert_eq!(workers.len(), 3);
    assert_eq!(hashes.len(), 1);
    assert_eq!(metrics["safetensors_tensor"], "source.weight");
    assert_eq!(metrics["gguf_open_strategy"], "eager_file");
    #[cfg(target_os = "linux")]
    assert!(
        metrics.contains_key("qif_peak_rss_kb") && metrics.contains_key("qif_verify_peak_rss_kb")
    );
    fs::remove_dir_all(root).unwrap();
}
