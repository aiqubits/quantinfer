#![cfg(feature = "cpu")]

#[path = "support/qif_layout_fixture.rs"]
mod fixture;

use quantinfer::{
    cpu::{resolve_backend, CpuModel, ModelSource},
    format::{
        qif::{self, Codec},
        GgufReader,
    },
    runtime::DecoderModel,
};
use std::fs;

#[test]
fn both_dense_adapters_preserve_layouts_profiles_and_dimension_order() {
    let root = std::env::temp_dir().join(format!(
        "quantinfer-layout-regression-{}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    let mut failures = Vec::new();
    for architecture in ["qwen3", "llama"] {
        let source = fixture::model(architecture);
        let reference = CpuModel::from_gguf(&source, None, None).unwrap();
        let mut cache = reference.new_cache(16).unwrap();
        let prefill = reference.forward(&[1, 2, 3], &mut cache).unwrap();
        let decode = reference.forward(&[4], &mut cache).unwrap();
        for ggml in [false, true] {
            for mixed in [false, true] {
                let p = fixture::publication(&source, architecture, ggml, mixed);
                for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
                    for kind in ["owned", "file", "package"] {
                        let label =
                            format!("{architecture}-ggml{ggml}-mixed{mixed}-{codec:?}-{kind}");
                        let path = root.join(&label);
                        let reader = if kind == "package" {
                            qif::write_qif_package_with_codec(
                                p.manifest.clone(),
                                &p.metadata,
                                |name| Ok(p.data[name].as_slice()),
                                &path,
                                Some(4096),
                                codec,
                            )
                            .unwrap();
                            let mut native = qif::PackageReader::open(&path).unwrap();
                            native.verify().unwrap();
                            ModelSource::open(&path).unwrap().open_reader(None).unwrap()
                        } else {
                            let mut bytes = Vec::new();
                            qif::write_qif_with_codec(
                                p.manifest.clone(),
                                &p.metadata,
                                |name| Ok(p.data[name].as_slice()),
                                &mut bytes,
                                codec,
                            )
                            .unwrap();
                            qif::Reader::open(&bytes).unwrap().verify().unwrap();
                            if kind == "owned" {
                                GgufReader::open_artifact(bytes).unwrap()
                            } else {
                                fs::write(&path, bytes).unwrap();
                                ModelSource::open(&path).unwrap().open_reader(None).unwrap()
                            }
                        };
                        for direct in [false, true] {
                            let model = if direct {
                                CpuModel::with_backend(
                                    resolve_backend(&source, None).unwrap(),
                                    &reader,
                                    None,
                                )
                            } else {
                                CpuModel::from_gguf(&reader, None, None)
                            };
                            match model {
                                Err(error) => {
                                    failures.push(format!("{label}/direct{direct}: {error}"))
                                }
                                Ok(model) => {
                                    let mut cache = model.new_cache(16).unwrap();
                                    for (tokens, expected, length) in
                                        [(&[1, 2, 3][..], &prefill, 3), (&[4][..], &decode, 4)]
                                    {
                                        let actual = model.forward(tokens, &mut cache).unwrap();
                                        let max = actual
                                            .as_f32()
                                            .unwrap()
                                            .iter()
                                            .zip(expected.as_f32().unwrap())
                                            .map(|(a, b)| (a - b).abs())
                                            .fold(0.0_f32, f32::max);
                                        if max > 1e-6 {
                                            failures.push(format!(
                                                "{label}/direct{direct}: logit error {max}"
                                            ));
                                        }
                                        assert_eq!(cache.sequence_length(0).unwrap(), length);
                                    }
                                    assert_eq!(reader.cached_tensor_bytes(), 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fs::remove_dir_all(root).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
