use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{
    qif::{write_qif, Reader},
    GgufReader, ModelManifest, TensorManifest,
};
use quantinfer_runtime::{LlamaModel, Qwen3Model};
use std::collections::BTreeMap;

#[test]
fn both_dense_loaders_reject_a_known_but_different_backend_before_reading_weights() {
    for architecture in ["qwen3", "llama"] {
        let mut manifest = ModelManifest::new("requirements", architecture);
        let mut descriptor =
            TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2])).unwrap();
        descriptor.backend_requirements.push("cuda".into());
        manifest.tensors.push(TensorManifest {
            name: "weight".into(),
            descriptor,
            checksum_sha256: None,
        });
        let mut bytes = Vec::new();
        write_qif(manifest, &BTreeMap::new(), |_| Ok(&[0; 8]), &mut bytes).unwrap();
        let reader = GgufReader::open_artifact(bytes).unwrap();
        let error = match architecture {
            "qwen3" => Qwen3Model::from_gguf(CpuScalar::default(), &reader)
                .err()
                .unwrap(),
            _ => LlamaModel::from_gguf(CpuScalar::default(), &reader)
                .err()
                .unwrap(),
        };
        assert!(
            error.to_string().contains("backend requirement cuda"),
            "{error}"
        );
    }
}

#[test]
fn native_qif_preserves_requirements_for_inspection_but_runtime_enforces_them() {
    for architecture in ["qwen3", "llama"] {
        for field in 0..4 {
            let mut manifest = ModelManifest::new("requirements", architecture);
            manifest.tensors.push(TensorManifest {
                name: "weight".into(),
                descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))
                    .unwrap(),
                checksum_sha256: None,
            });
            let expected = match field {
                0 => {
                    manifest.compatibility.minimum_quantinfer_version = Some("999.0.0".into());
                    "requires QuantInfer"
                }
                1 => {
                    manifest
                        .compatibility
                        .required_features
                        .push("future-feature".into());
                    "required feature"
                }
                2 => {
                    manifest.tensors[0]
                        .descriptor
                        .backend_requirements
                        .push("future-backend".into());
                    "backend requirement"
                }
                _ => {
                    manifest
                        .kernel_requirements
                        .operations
                        .push("future-operation".into());
                    "required operation"
                }
            };
            let mut bytes = Vec::new();
            write_qif(manifest, &BTreeMap::new(), |_| Ok(&[0; 8]), &mut bytes).unwrap();
            Reader::open(&bytes).unwrap().verify().unwrap();
            let reader = GgufReader::open_artifact(bytes).unwrap();
            let spec = match architecture {
                "qwen3" => Qwen3Model::<CpuScalar>::spec_from_gguf(&reader),
                _ => LlamaModel::<CpuScalar>::spec_from_gguf(&reader),
            };
            assert!(
                spec.unwrap_err().to_string().contains(expected),
                "{architecture} field {field}"
            );
        }
    }
}
