#![cfg(feature = "gptq")]

#[path = "support/calibrated_fixture.rs"]
mod fixture;

use quantinfer::{
    cpu::{CpuModel, ModelSource},
    runtime::DecoderModel,
};
use quantinfer_format::{qif, GgmlType, GgufReader};
use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn command(source: &Path, dataset: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
    command
        .arg("gptq")
        .arg("--input")
        .arg(source)
        .arg("--dataset")
        .arg(dataset)
        .arg("--output")
        .arg(output)
        .args(["--max-sequence-length", "16"]);
    command
}
fn success(result: Output) {
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
fn forward(path: &Path) -> Vec<f32> {
    let source = ModelSource::open(path).unwrap();
    let reader = source.open_reader(None).unwrap();
    let model = CpuModel::from_gguf(&reader, source.package().map(|p| &p.manifest), None).unwrap();
    model
        .forward(&[1, 3, 2], &mut model.new_cache(16).unwrap())
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec()
}

#[test]
fn quantizes_both_architectures_and_dimension_profiles_through_all_publication_options() {
    let root = tempfile::tempdir().unwrap();
    let dataset = root.path().join("calibration.jsonl");
    fs::write(&dataset, "\"a b\"\n{\"text\":\"b a a\"}\n").unwrap();
    let holdout = root.path().join("holdout.jsonl");
    fs::write(&holdout, "\"b b\"\n").unwrap();
    for architecture in ["qwen3", "llama"] {
        let native = root.path().join(format!("{architecture}-native.gguf"));
        fs::write(&native, fixture::model(architecture, false)).unwrap();
        let expected = forward(&native);
        for ggml in [false, true] {
            let input = root.path().join(format!("{architecture}-{ggml}.gguf"));
            let original = fixture::model(architecture, ggml);
            fs::write(&input, &original).unwrap();
            let actual = forward(&input);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(
                error < 1e-6,
                "source profiles differ: {architecture}/{ggml}/{error}"
            );
            let source = GgufReader::open(original.clone()).unwrap();
            for bits in [4, 8] {
                for profile in ["single", "package"] {
                    for codec in ["none", "zstd", "lz4"] {
                        let label = format!("{architecture}-{ggml}-{bits}-{profile}-{codec}");
                        let output = root.path().join(&label);
                        let report = root.path().join(format!("{label}.json"));
                        let mut cmd = command(&input, &dataset, &output);
                        cmd.args([
                            "--bits",
                            &bits.to_string(),
                            "--qif-profile",
                            profile,
                            "--qif-codec",
                            codec,
                        ]);
                        cmd.arg("--validation-dataset")
                            .arg(&holdout)
                            .arg("--report")
                            .arg(&report);
                        if profile == "package" {
                            cmd.args(["--max-shard-bytes", "16384"]);
                        }
                        success(cmd.output().unwrap());
                        let report: Value =
                            serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
                        assert_eq!(report["quantized_projections"], 14);
                        assert_eq!(report["calibration_token_rows"], 8);
                        assert_eq!(report["quality_status"], "passed");
                        assert_eq!(report["artifact_parity"], true);
                        assert_eq!(
                            report["calibration_dataset_sha256"],
                            quantinfer_format::sha256_hex(&fs::read(&dataset).unwrap())
                        );
                        let published = ModelSource::open(&output)
                            .unwrap()
                            .open_reader(None)
                            .unwrap();
                        published.verify_qif().unwrap();
                        for tensor in published.tensors() {
                            if quantinfer_format::is_linear_projection_storage_name(&tensor.name) {
                                assert_eq!(
                                    tensor.dtype,
                                    if bits == 4 {
                                        GgmlType::Q4_0
                                    } else {
                                        GgmlType::Q8_0
                                    }
                                );
                                let record = report["projections"]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .find(|r| r["tensor"] == tensor.name)
                                    .unwrap();
                                assert_eq!(record["calibration_rows"], 8);
                                assert_eq!(
                                    record["payload_sha256"],
                                    quantinfer_format::sha256_hex(
                                        published.tensor_data(&tensor.name).unwrap().as_ref()
                                    )
                                );
                            } else {
                                assert_eq!(
                                    published.tensor_data(&tensor.name).unwrap().as_ref(),
                                    source.tensor_bytes(&tensor.name).unwrap()
                                );
                            }
                        }
                        assert_eq!(fs::read(&input).unwrap(), original);
                        let requantized = root.path().join(format!("{label}-second"));
                        let result = command(&output, &dataset, &requantized).output().unwrap();
                        assert!(!result.status.success());
                        assert!(
                            String::from_utf8_lossy(&result.stderr).contains("already quantized")
                        );
                        assert!(!requantized.exists());
                    }
                }
            }
        }
    }
}

#[test]
fn native_sources_keep_typed_hints_and_source_revisions() {
    let root = tempfile::tempdir().unwrap();
    let source = GgufReader::open(fixture::model("llama", false)).unwrap();
    let (mut manifest, mut metadata) = qif::describe_gguf(&source, "native-source").unwrap();
    manifest.source_revision = Some("fixed-source-revision".into());
    manifest.tokenizer_revision = Some("fixed-tokenizer-revision".into());
    metadata.insert(
        "optional.hints".into(),
        qif::MetadataValue::Map(std::collections::BTreeMap::from([(
            "batch".into(),
            qif::MetadataValue::U64(7),
        )])),
    );
    let dataset = root.path().join("data.jsonl");
    fs::write(&dataset, "\"a b\"\n").unwrap();
    for package in [false, true] {
        let input = root.path().join(format!("source-{package}"));
        if package {
            qif::write_qif_package_with_codec(
                manifest.clone(),
                &metadata,
                |n| source.tensor_bytes(n),
                &input,
                Some(16384),
                qif::Codec::Lz4,
            )
            .unwrap();
        } else {
            qif::write_qif_with_codec(
                manifest.clone(),
                &metadata,
                |n| source.tensor_bytes(n),
                &mut fs::File::create(&input).unwrap(),
                qif::Codec::Zstd,
            )
            .unwrap();
        }
        let output = root.path().join(format!("result-{package}"));
        success(command(&input, &dataset, &output).output().unwrap());
        let reader = ModelSource::open(&output)
            .unwrap()
            .open_reader(None)
            .unwrap();
        reader.verify_qif().unwrap();
        assert_eq!(
            reader.qif_metadata().unwrap()["optional.hints"],
            metadata["optional.hints"]
        );
        let produced = reader.embedded_manifest().unwrap();
        assert_eq!(produced.source_revision, manifest.source_revision);
        assert_eq!(produced.tokenizer_revision, manifest.tokenizer_revision);
        assert_eq!(produced.source_format, "qif");
    }
}

#[test]
fn verified_model_package_retains_provenance_and_rejects_a_changed_source() {
    let root = tempfile::tempdir().unwrap();
    let package = root.path().join("source");
    fs::create_dir(&package).unwrap();
    let bytes = fixture::model("qwen3", false);
    let reader = GgufReader::open(bytes.clone()).unwrap();
    let (mut manifest, _) = qif::describe_gguf(&reader, "verified-package").unwrap();
    manifest.source_revision = Some("source-revision".into());
    manifest.tokenizer_revision = Some("tokenizer-revision".into());
    manifest
        .checksums
        .insert("model.gguf".into(), quantinfer_format::sha256_hex(&bytes));
    fs::write(package.join("model.gguf"), bytes).unwrap();
    fs::write(
        package.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let dataset = root.path().join("calibration.jsonl");
    fs::write(&dataset, "\"a b\"\n").unwrap();
    let output = root.path().join("model.qif");
    success(command(&package, &dataset, &output).output().unwrap());
    let produced = ModelSource::open(output)
        .unwrap()
        .open_reader(None)
        .unwrap();
    let actual = produced.embedded_manifest().unwrap();
    assert_eq!(actual.source_revision, manifest.source_revision);
    assert_eq!(actual.tokenizer_revision, manifest.tokenizer_revision);
    assert!(actual.checksums.is_empty());
    fs::write(package.join("model.gguf"), b"changed artifact").unwrap();
    let rejected = root.path().join("rejected.qif");
    assert!(!command(&package, &dataset, &rejected)
        .output()
        .unwrap()
        .status
        .success());
    assert!(!rejected.exists());
}

#[test]
fn floating_half_sources_keep_nonprojection_payloads_in_their_original_precision() {
    let root = tempfile::tempdir().unwrap();
    let source = GgufReader::open(fixture::model("llama", false)).unwrap();
    let dataset = root.path().join("calibration.jsonl");
    fs::write(&dataset, "\"a b\"\n").unwrap();
    for bf16 in [false, true] {
        let mut writer = quantinfer_format::GgufWriter::new();
        for (key, value) in source.metadata() {
            writer.insert_metadata(key, value.clone());
        }
        for tensor in source.tensors() {
            let decoded = source.tensor_f32(&tensor.name).unwrap();
            if bf16 {
                writer
                    .add_bf16_tensor(
                        &tensor.name,
                        tensor.shape.clone(),
                        decoded.as_f32().unwrap(),
                    )
                    .unwrap();
            } else {
                writer
                    .add_f16_tensor(
                        &tensor.name,
                        tensor.shape.clone(),
                        decoded.as_f32().unwrap(),
                    )
                    .unwrap();
            }
        }
        let bytes = writer.finish().unwrap();
        let input = root.path().join(format!("half-{bf16}.gguf"));
        fs::write(&input, &bytes).unwrap();
        let original = GgufReader::open(bytes).unwrap();
        let output = root.path().join(format!("model-{bf16}.qif"));
        success(command(&input, &dataset, &output).output().unwrap());
        let actual = ModelSource::open(&output)
            .unwrap()
            .open_reader(None)
            .unwrap();
        actual.verify_qif().unwrap();
        for tensor in actual.tensors() {
            if !quantinfer_format::is_linear_projection_storage_name(&tensor.name) {
                assert_eq!(
                    tensor.dtype,
                    if bf16 { GgmlType::BF16 } else { GgmlType::F16 }
                );
                assert_eq!(
                    actual.tensor_data(&tensor.name).unwrap().as_ref(),
                    original.tensor_bytes(&tensor.name).unwrap()
                );
            }
        }
    }
}

#[test]
fn quality_and_configuration_failures_publish_no_artifact() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("source.gguf");
    fs::write(&input, fixture::model("llama", false)).unwrap();
    let dataset = root.path().join("data.jsonl");
    fs::write(&dataset, "\"a b\"\n").unwrap();
    let output = root.path().join("output");
    let cases: &[&[&str]] = &[
        &["--bits", "2"],
        &["--group-size", "3"],
        &["--damping", "NaN"],
        &["--maximum-workspace-bytes", "1"],
        &["--maximum-statistics-bytes", "1"],
        &["--bits", "4", "--bits", "8"],
        &["--unknown"],
        &["--model-id", ""],
        &["--maximum-kl", "0.1"],
        &["--max-dataset-bytes", "1"],
    ];
    for args in cases {
        let result = command(&input, &dataset, &output)
            .args(*args)
            .output()
            .unwrap();
        assert!(!result.status.success(), "{args:?}");
        assert!(!output.exists());
    }
    let result = command(&input, &dataset, &output)
        .arg("--validation-dataset")
        .arg(&dataset)
        .args(["--maximum-kl", "1e-300"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("failed quality gates"));
    assert!(!output.exists());
    fs::write(&output, b"existing output").unwrap();
    assert!(!command(&input, &dataset, &output)
        .output()
        .unwrap()
        .status
        .success());
    assert_eq!(fs::read(&output).unwrap(), b"existing output");
}
