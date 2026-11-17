use std::collections::BTreeMap;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{
    qif::{write_qif, MetadataValue},
    GgufReader, ModelManifest, TensorManifest,
};
use quantinfer_runtime::{LlamaModel, Qwen3Model};

#[test]
fn native_maps_do_not_hide_invalid_model_parameters_behind_defaults() {
    for architecture in ["qwen3", "llama"] {
        let mut model = ModelManifest::new("metadata-fixture", architecture);
        model.tensors.push(TensorManifest {
            name: "token_embd.weight".into(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![16, 8]))
                .unwrap(),
            checksum_sha256: None,
        });
        let map = MetadataValue::Map(BTreeMap::from([(
            "prefetch".into(),
            MetadataValue::Bool(true),
        )]));
        let mut metadata = BTreeMap::from([("optional.hints".into(), map.clone())]);
        for (key, value) in [
            ("vocab_size", 16),
            ("embedding_length", 8),
            ("feed_forward_length", 16),
            ("block_count", 1),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.head_dim", 4),
            ("context_length", 32),
        ] {
            metadata.insert(format!("{architecture}.{key}"), MetadataValue::U32(value));
        }
        metadata.insert(
            format!("{architecture}.tie_word_embeddings"),
            MetadataValue::Bool(true),
        );
        let spec = |values: &BTreeMap<String, MetadataValue>| {
            let mut bytes = Vec::new();
            write_qif(model.clone(), values, |_| Ok(&[0; 512]), &mut bytes).unwrap();
            let reader = GgufReader::open_artifact(bytes).unwrap();
            match architecture {
                "qwen3" => Qwen3Model::<CpuScalar>::spec_from_gguf(&reader),
                _ => LlamaModel::<CpuScalar>::spec_from_gguf(&reader),
            }
        };
        // An unknown hint remains harmless; valid optional parameters may be absent.
        spec(&metadata).unwrap();
        for suffix in [
            "rope.freq_base",
            "attention.head_count_kv",
            "tie_word_embeddings",
        ] {
            let key = format!("{architecture}.{suffix}");
            let mut invalid = metadata.clone();
            invalid.insert(key.clone(), map.clone());
            let error = spec(&invalid)
                .expect_err("present but invalid parameters must not become defaults")
                .to_string();
            assert!(error.contains(&key), "{error}");
        }
    }
}
