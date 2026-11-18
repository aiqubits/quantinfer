use std::{collections::BTreeMap, fs, process::Command};

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{
    qif::{write_qif_package_with_codec, write_qif_with_codec, Codec, MetadataValue},
    GgufWriter, ModelManifest, TensorManifest,
};
use serde_json::Value;

fn invoke(command: &str, path: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_quantinfer-cli"))
        .args([command, "--json", "--input"])
        .arg(path)
        .output()
        .unwrap()
}

#[test]
fn native_inspect_preserves_typed_maps_shape_and_alias_without_verifying_payload() {
    let root = tempfile::tempdir().unwrap();
    let bytes = [0, 0, 128, 63, 0, 0, 0, 64];
    let metadata = BTreeMap::from([
        (
            "quantinfer.qif.ggml_dimension_order".into(),
            MetadataValue::Bool(true),
        ),
        (
            "optional.hints".into(),
            MetadataValue::Map(BTreeMap::from([
                ("small".into(), MetadataValue::U32(1)),
                ("wide".into(), MetadataValue::U64(u64::MAX)),
                (
                    "array".into(),
                    MetadataValue::Array(vec![MetadataValue::I64(-1)]),
                ),
            ])),
        ),
    ]);
    for (codec, codec_id) in [(Codec::None, 0), (Codec::Zstd, 1), (Codec::Lz4, 2)] {
        for package in [false, true] {
            let path = root.path().join(format!("{codec_id}-{package}"));
            let mut manifest = ModelManifest::new("native-inspect", "future-model");
            // Inspection must remain available without an executable runtime for
            // the artifact's architecture or mandatory capability requirements.
            manifest
                .compatibility
                .required_features
                .push("future-feature".into());
            for name in ["a", "b"] {
                manifest.tensors.push(TensorManifest {
                    name: name.into(),
                    descriptor: TensorStorageDescriptor::floating(
                        DType::F32,
                        Shape::from(vec![2, 1]),
                    )
                    .unwrap(),
                    checksum_sha256: None,
                });
            }
            if package {
                write_qif_package_with_codec(
                    manifest,
                    &metadata,
                    |_| Ok(&bytes),
                    &path,
                    Some(4096),
                    codec,
                )
                .unwrap();
            } else {
                write_qif_with_codec(
                    manifest,
                    &metadata,
                    |_| Ok(&bytes),
                    &mut fs::File::create(&path).unwrap(),
                    codec,
                )
                .unwrap();
            }
            let inspect = invoke("inspect", &path);
            assert!(
                inspect.status.success(),
                "{}",
                String::from_utf8_lossy(&inspect.stderr)
            );
            let report: Value = serde_json::from_slice(&inspect.stdout).unwrap();
            assert_eq!(report["format"], "QIF");
            assert_eq!(
                report["profile"],
                if package { "PACKAGE" } else { "SINGLE" }
            );
            assert_eq!(report["control_verified"], true);
            assert_eq!(report["payload_verified"], false);
            assert_eq!(report["publisher_trust"], "not_evaluated");
            assert_eq!(
                report["metadata"]["optional.hints"]["value"]["small"]["type"],
                "U32"
            );
            assert_eq!(
                report["metadata"]["optional.hints"]["value"]["wide"]["value"],
                u64::MAX
            );
            assert_eq!(
                report["metadata"]["optional.hints"]["value"]["array"]["value"][0]["value"],
                -1
            );
            assert_eq!(report["tensors"][0]["shape"], serde_json::json!([2, 1]));
            assert_eq!(report["tensors"][0]["codec"], codec_id);
            assert_eq!(report["tensors"][1]["alias_of"], 0);
            let verified = invoke("verify", &path);
            assert!(
                verified.status.success(),
                "{}",
                String::from_utf8_lossy(&verified.stderr)
            );
            let verified: Value = serde_json::from_slice(&verified.stdout).unwrap();
            assert_eq!(verified["payload_verified"], true);
            assert_eq!(verified["tensor_count"], 2);
            // Leave control hashes valid, but damage the physical payload.
            let payload = if package {
                path.join("shard-00000.qifd")
            } else {
                path.clone()
            };
            let mut stored = fs::read(&payload).unwrap();
            let offset = if package {
                0
            } else {
                u64::from_le_bytes(stored[72..80].try_into().unwrap()) as usize
            };
            stored[offset] ^= 0xff;
            fs::write(payload, stored).unwrap();
            assert!(invoke("inspect", &path).status.success());
            let failed = invoke("verify", &path);
            assert!(!failed.status.success());
            assert!(failed.stdout.is_empty());
            // Malformed controls must also fail inspection, even without verify.
            if !package {
                let mut stored = fs::read(&path).unwrap();
                stored[88] ^= 1;
                fs::write(&path, stored).unwrap();
                assert!(!invoke("inspect", &path).status.success());
            }
        }
    }
}

#[test]
fn native_commands_do_not_report_unchecksummed_gguf_as_verified() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("model.gguf");
    let mut writer = GgufWriter::new();
    writer
        .add_f32_tensor("weight", vec![2], &[1.0, 2.0])
        .unwrap();
    fs::write(&path, writer.finish().unwrap()).unwrap();
    assert!(!invoke("inspect", &path).status.success());
    assert!(!invoke("verify", &path).status.success());
    let compatible = invoke("inspect-gguf", &path);
    assert!(compatible.status.success());
    let report: Value = serde_json::from_slice(&compatible.stdout).unwrap();
    assert_eq!(report["tensors"][0]["dtype"], "F32");
    assert!(report.get("payload_verified").is_none());
}
