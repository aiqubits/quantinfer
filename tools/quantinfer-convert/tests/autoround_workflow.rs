#![cfg(feature = "autoround")]
#[path = "support/calibrated_fixture.rs"]
mod fixture;

use quantinfer::cpu::{CpuModel, ModelSource};
use quantinfer::runtime::DecoderModel;
use quantinfer_format::{sha256_hex, GgmlType, GgufReader};
use serde_json::Value;
use std::{fs, path::Path, process::Command};

fn command(source: &Path, dataset: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
    command
        .arg("autoround")
        .arg("--input")
        .arg(source)
        .arg("--dataset")
        .arg(dataset)
        .arg("--output")
        .arg(output)
        .args([
            "--max-sequence-length",
            "16",
            "--iterations",
            "8",
            "--threads",
            "2",
        ]);
    command
}

#[test]
fn autoround_publishes_optimized_codes_and_preserves_floating_tensors() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input.gguf");
    let dataset = root.path().join("calibration.jsonl");
    fs::write(&dataset, "\"a b\"\n{\"text\":\"b a a\"}\n").unwrap();
    let holdout = root.path().join("holdout.jsonl");
    fs::write(&holdout, "\"b b\"\n").unwrap();
    for architecture in ["qwen3", "llama"] {
        for ggml in [false, true] {
            let bytes = fixture::model(architecture, ggml);
            fs::write(&input, &bytes).unwrap();
            let source = GgufReader::open(bytes.clone()).unwrap();
            for bits in [4, 8] {
                let mut expected = None;
                for (profile, codec) in [("single", "none"), ("package", "zstd"), ("single", "lz4")]
                {
                    let output = root
                        .path()
                        .join(format!("{architecture}-{ggml}-{bits}-{profile}-{codec}"));
                    let report = output.with_extension("json");
                    let result = command(&input, &dataset, &output)
                        .args([
                            "--bits",
                            &bits.to_string(),
                            "--group-size",
                            "64",
                            "--qif-profile",
                            profile,
                            "--qif-codec",
                            codec,
                        ])
                        .arg("--validation-dataset")
                        .arg(&holdout)
                        .arg("--report")
                        .arg(&report)
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    assert_eq!(fs::read(&input).unwrap(), bytes);
                    let report: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
                    assert_eq!(report["quantized_projections"], 14);
                    assert_eq!(report["artifact_parity"], true);
                    assert_eq!(report["quality_status"], "passed");
                    let records = report["projections"].as_array().unwrap();
                    assert!(records.iter().all(|r| {
                        r["best_mse"].as_f64().unwrap() < r["initial_mse"].as_f64().unwrap()
                            && r["calibration_rows"] == 8
                    }));
                    let reader = ModelSource::open(&output)
                        .unwrap()
                        .open_reader(None)
                        .unwrap();
                    reader.verify_qif().unwrap();
                    for tensor in reader.tensors() {
                        let raw = reader.tensor_data(&tensor.name).unwrap();
                        if quantinfer_format::is_linear_projection_storage_name(&tensor.name) {
                            let dtype = if bits == 4 {
                                GgmlType::Q4_0
                            } else {
                                GgmlType::Q8_0
                            };
                            assert_eq!(tensor.dtype, dtype);
                            let record =
                                records.iter().find(|r| r["tensor"] == tensor.name).unwrap();
                            assert_eq!(record["payload_sha256"], sha256_hex(raw.as_ref()));
                        } else {
                            assert_eq!(raw.as_ref(), source.tensor_bytes(&tensor.name).unwrap());
                        }
                    }
                    let model = CpuModel::from_gguf(&reader, None, None).unwrap();
                    let logits = model
                        .forward(&[1, 3, 2], &mut model.new_cache(16).unwrap())
                        .unwrap();
                    if let Some(expected) = expected.as_ref() {
                        assert_eq!(&logits, expected);
                    } else {
                        expected = Some(logits);
                    }
                }
            }
        }
    }
}

#[test]
fn autoround_rejects_invalid_budget_without_publishing() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("source.gguf");
    let dataset = root.path().join("calibration.jsonl");
    let output = root.path().join("output.qif");
    fs::write(&input, fixture::model("qwen3", false)).unwrap();
    fs::write(&dataset, "\"a b\"\n").unwrap();
    let result = command(&input, &dataset, &output)
        .args(["--maximum-workspace-bytes", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("workspace budget"));
    assert!(!output.exists());
}
