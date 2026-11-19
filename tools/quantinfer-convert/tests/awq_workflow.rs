#![cfg(feature = "awq")]

#[path = "support/calibrated_fixture.rs"]
mod fixture;

use quantinfer::{
    cpu::{CpuModel, ModelSource},
    runtime::DecoderModel,
};
use quantinfer_format::{sha256_hex, GgmlType, GgufReader};
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn command(source: &Path, dataset: &Path, output: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
    cmd.arg("awq")
        .arg("--input")
        .arg(source)
        .arg("--dataset")
        .arg(dataset)
        .arg("--output")
        .arg(output)
        .args(["--max-sequence-length", "16", "--grid-points", "5"]);
    cmd
}

fn succeed(result: Output) {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn check_artifact(path: &Path, original: &GgufReader, report: &Value, bits: usize) -> Vec<f32> {
    let reader = ModelSource::open(path).unwrap().open_reader(None).unwrap();
    reader.verify_qif().unwrap();
    assert_eq!(report["quantized_projections"], 14);
    assert_eq!(report["normalizations"].as_array().unwrap().len(), 4);
    assert_eq!(report["quality_status"], "passed");
    assert_eq!(report["artifact_parity"], true);
    let clipping = report["clipping_searches"].as_array().unwrap();
    assert_eq!(clipping.len(), 10);
    for row in clipping {
        assert!(
            row["group_loss_after"].as_f64().unwrap() <= row["group_loss_before"].as_f64().unwrap()
        );
        assert_eq!(row["rows"], 8);
        assert_ne!(row["role"], "QueryProjection");
        assert_ne!(row["role"], "KeyProjection");
    }
    for t in reader.tensors() {
        let raw = reader.tensor_data(&t.name).unwrap();
        let record = report["projections"]
            .as_array()
            .unwrap()
            .iter()
            .chain(report["normalizations"].as_array().unwrap())
            .find(|r| r["tensor"] == t.name);
        if let Some(r) = record {
            assert_eq!(r["payload_sha256"], sha256_hex(raw.as_ref()));
            let dtype = if quantinfer_format::is_linear_projection_storage_name(&t.name) {
                if bits == 4 {
                    GgmlType::Q4_0
                } else {
                    GgmlType::Q8_0
                }
            } else {
                GgmlType::F32
            };
            assert_eq!(t.dtype, dtype);
        } else {
            assert_eq!(raw.as_ref(), original.tensor_bytes(&t.name).unwrap());
        }
    }
    let model = CpuModel::from_gguf(&reader, None, None).unwrap();
    model
        .forward(&[1, 3, 2], &mut model.new_cache(16).unwrap())
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec()
}

#[test]
fn awq_exports_real_layer_scales_and_clipping_in_both_layouts() {
    let root = tempfile::tempdir().unwrap();
    let dataset = root.path().join("calibration.jsonl");
    fs::write(&dataset, "\"a b\"\n{\"text\":\"b a a\"}\n").unwrap();
    let holdout = root.path().join("holdout.jsonl");
    fs::write(&holdout, "\"b b\"\n").unwrap();
    for (architecture, kv_heads) in [("qwen3", 2), ("llama", 8)] {
        for bits in [4, 8] {
            let mut expected = None;
            for ggml in [false, true] {
                let original = fixture::model_with_kv_heads(architecture, ggml, kv_heads);
                let source = root.path().join("source.gguf");
                fs::write(&source, &original).unwrap();
                let original_reader = GgufReader::open(original.clone()).unwrap();
                for profile in ["single", "package"] {
                    for codec in ["none", "zstd", "lz4"] {
                        let output = root
                            .path()
                            .join(format!("{architecture}-{bits}-{ggml}-{profile}-{codec}"));
                        let report = output.with_extension("json");
                        succeed(
                            command(&source, &dataset, &output)
                                .args([
                                    "--bits",
                                    &bits.to_string(),
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
                                .unwrap(),
                        );
                        let report: Value =
                            serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
                        let scales = report["scale_searches"].as_array().unwrap();
                        assert_eq!(scales.len(), if kv_heads == 2 { 6 } else { 8 });
                        for row in scales {
                            let losses = row["candidate_losses"].as_array().unwrap();
                            assert_eq!(losses.len(), 5);
                            let min = losses
                                .iter()
                                .map(|n| n.as_f64().unwrap())
                                .fold(f64::INFINITY, f64::min);
                            assert_eq!(row["loss"].as_f64().unwrap(), min);
                        }
                        let logits = check_artifact(&output, &original_reader, &report, bits);
                        if let Some(ref expected) = expected {
                            assert_eq!(
                                &logits, expected,
                                "{architecture}/{bits}/{ggml}/{profile}/{codec}"
                            );
                        } else {
                            expected = Some(logits);
                        }
                        assert_eq!(fs::read(&source).unwrap(), original);
                    }
                }
            }
        }
    }
}

#[test]
fn awq_budget_and_quality_failures_leave_no_output() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("model.gguf");
    let calibration = root.path().join("calibration.jsonl");
    let output = root.path().join("model.qif");
    let report = root.path().join("report.json");
    let holdout = root.path().join("holdout.jsonl");
    fs::write(&input, fixture::model("llama", false)).unwrap();
    fs::write(&calibration, "\"a b\"\n").unwrap();
    fs::write(&holdout, "\"b b\"\n").unwrap();
    for (arguments, message) in [
        (
            vec!["--maximum-statistics-bytes", "1"],
            "above the statistics budget",
        ),
        (vec!["--maximum-workspace-bytes", "1"], "workspace"),
        (
            vec![
                "--validation-dataset",
                holdout.to_str().unwrap(),
                "--maximum-kl",
                "1e-12",
            ],
            "quality gate",
        ),
    ] {
        let status = command(&input, &calibration, &output)
            .args(arguments)
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap();
        assert!(!status.status.success());
        assert!(
            String::from_utf8_lossy(&status.stderr).contains(message),
            "{:?}",
            status.stderr
        );
        assert!(!output.exists());
        assert!(!report.exists());
    }
}

#[test]
fn awq_folds_half_precision_norms_and_keeps_other_native_payloads() {
    use quantinfer_format::{qif, GgufWriter};
    let root = tempfile::tempdir().unwrap();
    let source = GgufReader::open(fixture::model("llama", false)).unwrap();
    let mut writer = GgufWriter::new();
    for (key, value) in source.metadata() {
        writer.insert_metadata(key, value.clone());
    }
    for tensor in source.tensors() {
        let weight = source.tensor_f32(&tensor.name).unwrap();
        let values = weight.as_f32().unwrap();
        if tensor.name.ends_with("attn_norm.weight") {
            writer
                .add_f16_tensor(&tensor.name, tensor.shape.clone(), values)
                .unwrap();
        } else if tensor.name.ends_with("ffn_norm.weight") || tensor.name == "output_norm.weight" {
            writer
                .add_bf16_tensor(&tensor.name, tensor.shape.clone(), values)
                .unwrap();
        } else {
            writer
                .add_f32_tensor(&tensor.name, tensor.shape.clone(), values)
                .unwrap();
        }
    }
    let source = GgufReader::open(writer.finish().unwrap()).unwrap();
    let (mut manifest, mut metadata) = qif::describe_gguf(&source, "half-norms").unwrap();
    manifest.source_revision = Some("pinned-half-source".into());
    let hint = qif::MetadataValue::Map(std::collections::BTreeMap::from([(
        "threads".into(),
        qif::MetadataValue::U32(2),
    )]));
    metadata.insert("optional.hints".into(), hint.clone());
    let input = root.path().join("half-package");
    qif::write_qif_package_with_codec(
        manifest,
        &metadata,
        |n| source.tensor_bytes(n),
        &input,
        None,
        qif::Codec::Lz4,
    )
    .unwrap();
    let dataset = root.path().join("prompts.jsonl");
    fs::write(&dataset, "\"a b\"\n\"b a a\"\n").unwrap();
    let output = root.path().join("output.qif");
    let report = root.path().join("report.json");
    succeed(
        command(&input, &dataset, &output)
            .arg("--no-clipping")
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap(),
    );
    let reader = ModelSource::open(&output)
        .unwrap()
        .open_reader(None)
        .unwrap();
    let report: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
    assert_eq!(report["quality_status"], "not_evaluated");
    assert_eq!(report["artifact_parity"], true);
    assert!(report["clipping_searches"].as_array().unwrap().is_empty());
    assert_eq!(
        reader
            .embedded_manifest()
            .unwrap()
            .source_revision
            .as_deref(),
        Some("pinned-half-source")
    );
    assert_eq!(reader.qif_metadata().unwrap()["optional.hints"], hint);
    for row in report["normalizations"].as_array().unwrap() {
        let name = row["tensor"].as_str().unwrap();
        let source_dtype = source.tensor_info(name).unwrap().dtype;
        assert!(matches!(source_dtype, GgmlType::F16 | GgmlType::BF16));
        assert_eq!(reader.tensor_info(name).unwrap().dtype, GgmlType::F32);
    }
    assert_eq!(
        reader.tensor_info("output_norm.weight").unwrap().dtype,
        GgmlType::BF16
    );
    assert_eq!(
        reader.tensor_data("output_norm.weight").unwrap().as_ref(),
        source.tensor_bytes("output_norm.weight").unwrap()
    );
}
