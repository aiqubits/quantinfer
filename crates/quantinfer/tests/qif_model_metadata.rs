#![cfg(feature = "cpu")]

#[path = "support/qif_layout_fixture.rs"]
mod fixture;

use quantinfer::{
    cpu::{estimated_weight_bytes, resolve_backend, validate_memory_budget, CpuModel, ModelSource},
    format::{qif, qif::MetadataValue, GgufReader},
    runtime::DecoderModel,
};
use std::{collections::BTreeMap, fs, path::PathBuf};

struct Root(PathBuf);
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn native_cpu_budget_counts_file_buffers_and_layout_workspace() {
    let root =
        Root(std::env::temp_dir().join(format!("quantinfer-model-budget-{}", std::process::id())));
    fs::create_dir(&root.0).unwrap();
    let source = fixture::model("llama");
    for mixed_layout in [false, true] {
        let artifact = fixture::publication(&source, "llama", false, mixed_layout);
        let file = root.0.join(format!("layout-{mixed_layout}.qif"));
        qif::write_qif(
            artifact.manifest,
            &artifact.metadata,
            |name| Ok(artifact.data[name].as_slice()),
            &mut fs::File::create(&file).unwrap(),
        )
        .unwrap();
        let reader = ModelSource::open(&file).unwrap().open_reader(None).unwrap();
        assert!(reader.is_file_backed());
        let spec = CpuModel::spec_from_gguf(&reader, None).unwrap();
        let base = reader.loading_memory_bytes().unwrap()
            + estimated_weight_bytes(&reader, &spec).unwrap();
        // The largest floating output-row matrix in this fixture is 8 × 16.
        let budget = base + if mixed_layout { 512 } else { 0 };
        validate_memory_budget(&reader, &spec, 12, 1, budget).unwrap();
        let error = validate_memory_budget(&reader, &spec, 12, 1, budget - 1).unwrap_err();
        assert!(error
            .to_string()
            .contains(&format!("estimate is {budget} bytes")));
    }
}

#[test]
fn model_parameter_validation_is_independent_of_container_and_dimension_profile() {
    let root = Root(
        std::env::temp_dir().join(format!("quantinfer-model-metadata-{}", std::process::id())),
    );
    fs::create_dir(&root.0).unwrap();
    let mut failures = Vec::new();
    let mut accepted = 0;
    let mut rejected = 0;
    for architecture in ["qwen3", "llama"] {
        let source = fixture::model(architecture);
        let baseline = CpuModel::from_gguf(&source, None, None).unwrap();
        let expected = baseline
            .forward(&[1, 2, 3], &mut baseline.new_cache(16).unwrap())
            .unwrap();
        for ggml in [false, true] {
            for (label, fields, supported) in [
                ("absent", vec![], true),
                (
                    "supported",
                    vec![
                        ("rope.scaling.type", MetadataValue::String("none".into())),
                        ("rope.dimension_count", MetadataValue::U32(4)),
                        ("attention.value_length", MetadataValue::U64(4)),
                        ("attention.sliding_window", MetadataValue::U32(0)),
                        ("optional.hint", MetadataValue::Map(BTreeMap::new())),
                    ],
                    true,
                ),
                (
                    "linear-scaling",
                    vec![
                        ("rope.scaling.type", MetadataValue::String("linear".into())),
                        ("rope.scaling.factor", MetadataValue::F32(2.0)),
                    ],
                    false,
                ),
                (
                    "sliding-window",
                    vec![("attention.sliding_window", MetadataValue::U32(2))],
                    false,
                ),
                (
                    "sliding-window-map",
                    vec![(
                        "attention.sliding_window",
                        MetadataValue::Map(BTreeMap::new()),
                    )],
                    false,
                ),
                (
                    "partial-rope",
                    vec![("rope.dimension_count", MetadataValue::U32(2))],
                    false,
                ),
                (
                    "value-width",
                    vec![("attention.value_length", MetadataValue::U32(3))],
                    false,
                ),
                (
                    "negative-dimension",
                    vec![("rope.dimension_count", MetadataValue::I32(-1))],
                    false,
                ),
                (
                    "string-width",
                    vec![("attention.value_length", MetadataValue::String("4".into()))],
                    false,
                ),
                (
                    "map-scaling",
                    vec![("rope.scaling.type", MetadataValue::Map(BTreeMap::new()))],
                    false,
                ),
                (
                    "map-dimension",
                    vec![("rope.dimension_count", MetadataValue::Map(BTreeMap::new()))],
                    false,
                ),
                (
                    "map-value",
                    vec![(
                        "attention.value_length",
                        MetadataValue::Map(BTreeMap::new()),
                    )],
                    false,
                ),
            ] {
                let mut p = fixture::publication(&source, architecture, ggml, false);
                for (suffix, value) in fields {
                    p.metadata.insert(format!("{architecture}.{suffix}"), value);
                }
                let mut bytes = Vec::new();
                qif::write_qif(
                    p.manifest.clone(),
                    &p.metadata,
                    |name| Ok(p.data[name].as_slice()),
                    &mut bytes,
                )
                .unwrap();
                // These are valid containers; unsupported execution requirements
                // are rejected by the model boundary rather than by hash checks.
                qif::Reader::open(&bytes).unwrap().verify().unwrap();
                for kind in ["owned", "file", "package"] {
                    let path = root.0.join(format!("{architecture}-{ggml}-{label}-{kind}"));
                    let reader = match kind {
                        "owned" => GgufReader::open_artifact(bytes.clone()).unwrap(),
                        "file" => {
                            fs::write(&path, &bytes).unwrap();
                            ModelSource::open(&path).unwrap().open_reader(None).unwrap()
                        }
                        "package" => {
                            qif::write_qif_package(
                                p.manifest.clone(),
                                &p.metadata,
                                |name| Ok(p.data[name].as_slice()),
                                &path,
                                Some(4096),
                            )
                            .unwrap();
                            ModelSource::open(&path).unwrap().open_reader(None).unwrap()
                        }
                        _ => unreachable!(),
                    };
                    if kind != "owned" {
                        reader.verify_qif().unwrap();
                    }
                    assert_eq!(
                        CpuModel::spec_from_gguf(&reader, None).is_ok(),
                        supported,
                        "{architecture}/{ggml}/{label}/{kind} spec"
                    );
                    for direct in [false, true] {
                        let result = if direct {
                            CpuModel::with_backend(
                                resolve_backend(&source, None).unwrap(),
                                &reader,
                                None,
                            )
                        } else {
                            CpuModel::from_gguf(&reader, None, None)
                        };
                        let case = format!("{architecture}/{ggml}/{label}/{kind}/{direct}");
                        match result {
                            Ok(model) if supported => {
                                let logits = model
                                    .forward(&[1, 2, 3], &mut model.new_cache(16).unwrap())
                                    .unwrap();
                                for (a, b) in logits
                                    .as_f32()
                                    .unwrap()
                                    .iter()
                                    .zip(expected.as_f32().unwrap())
                                {
                                    assert!((a - b).abs() <= 1e-6, "{case}");
                                }
                                accepted += 1;
                            }
                            Err(_) if !supported => rejected += 1,
                            other => failures.push(format!(
                                "{case}: expected supported={supported}, accepted={}",
                                other.is_ok()
                            )),
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!((accepted, rejected), (48, 240));
}
