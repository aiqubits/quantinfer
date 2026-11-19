#![cfg(feature = "cpu")]
use quantinfer::{
    cpu::{CpuModel, ModelSource},
    format::{
        qif::{import_gguf_package, import_gguf_with_codec, write_qif, Codec, Reader},
        GgufReader, GgufValue, GgufWriter,
    },
    runtime::DecoderModel,
};
use std::{collections::BTreeMap, fs};

fn model() -> GgufReader {
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("qwen3".into()));
    for (name, value) in [
        ("vocab_size", 16),
        ("context_length", 32),
        ("embedding_length", 8),
        ("feed_forward_length", 16),
        ("block_count", 1),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.head_dim", 4),
    ] {
        writer.insert_metadata(format!("qwen3.{name}"), GgufValue::U32(value));
    }
    writer.insert_metadata("qwen3.tie_word_embeddings", GgufValue::Bool(true));
    let shapes = BTreeMap::from([
        ("token_embd.weight", vec![16, 8]),
        ("output_norm.weight", vec![8]),
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.attn_q.weight", vec![8, 8]),
        ("blk.0.attn_q_norm.weight", vec![4]),
        ("blk.0.attn_k.weight", vec![8, 4]),
        ("blk.0.attn_k_norm.weight", vec![4]),
        ("blk.0.attn_v.weight", vec![8, 4]),
        ("blk.0.attn_output.weight", vec![8, 8]),
        ("blk.0.ffn_norm.weight", vec![8]),
        ("blk.0.ffn_gate.weight", vec![8, 16]),
        ("blk.0.ffn_up.weight", vec![8, 16]),
        ("blk.0.ffn_down.weight", vec![16, 8]),
    ]);
    for (name, shape) in shapes {
        let data: Vec<_> = (0..shape.iter().product())
            .map(|index| {
                if name.contains("norm") {
                    1.0
                } else {
                    // Avoid symmetric square matrices hiding transposition errors.
                    ((index * 17 + index / 3 + 5) % 29) as f32 * 0.03 - 0.4
                }
            })
            .collect();
        writer.add_f32_tensor(name, shape, &data).unwrap();
    }
    GgufReader::open(writer.finish().unwrap()).unwrap()
}

#[test]
fn default_model_source_loads_both_native_profiles_without_retaining_raw_weights() {
    let original = model();
    let expected = CpuModel::from_gguf(&original, None, None).unwrap();
    let logits = expected
        .forward(&[1, 2], &mut expected.new_cache(16).unwrap())
        .unwrap();
    let mut required_bytes = Vec::new();
    import_gguf_with_codec(&original, "required", &mut required_bytes, Codec::None).unwrap();
    let native = Reader::open(&required_bytes).unwrap();
    let mut required = native.manifest().clone();
    required.compatibility.required_features = quantinfer::runtime::RUNTIME_FEATURES
        .iter()
        .map(|name| (*name).into())
        .collect();
    required.tensors[0].descriptor.backend_requirements = vec!["cpu".into()];
    let payloads: BTreeMap<_, _> = native
        .tensors()
        .iter()
        .map(|info| (info.name.clone(), native.tensor_data(&info.name).unwrap()))
        .collect();
    let mut bytes = Vec::new();
    write_qif(
        required.clone(),
        native.metadata(),
        |name| Ok(payloads[name].as_slice()),
        &mut bytes,
    )
    .unwrap();
    let supported = GgufReader::open_artifact(bytes).unwrap();
    let loaded = CpuModel::from_gguf(&supported, None, None).unwrap();
    let actual = loaded
        .forward(&[1, 2], &mut loaded.new_cache(16).unwrap())
        .unwrap();
    assert_eq!(actual.as_f32().unwrap(), logits.as_f32().unwrap());
    required.tensors[0].descriptor.backend_requirements = vec!["cuda".into()];
    assert!(CpuModel::from_gguf(&supported, Some(&required), None)
        .err()
        .unwrap()
        .to_string()
        .contains("backend requirement cuda"));
    let root = std::env::temp_dir().join(format!("native-runtime-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
        for package in [false, true] {
            let path = root.join(format!("{codec:?}-{package}"));
            if package {
                import_gguf_package(&original, "fixture", &path, Some(4096), codec).unwrap();
            } else {
                import_gguf_with_codec(
                    &original,
                    "fixture",
                    &mut fs::File::create(&path).unwrap(),
                    codec,
                )
                .unwrap();
            }
            let source = ModelSource::open(&path).unwrap();
            assert!(source.open_reader(Some(1)).is_err());
            let reader = source.open_reader(None).unwrap();
            assert!(reader.is_file_backed());
            assert_eq!(reader.cached_tensor_bytes(), 0);
            let loaded = CpuModel::from_gguf(&reader, None, None).unwrap();
            assert_eq!(reader.cached_tensor_bytes(), 0);
            let actual = loaded
                .forward(&[1, 2], &mut loaded.new_cache(16).unwrap())
                .unwrap();
            assert_eq!(actual.as_f32().unwrap(), logits.as_f32().unwrap());
        }
    }
    let ambiguous = root.join("ambiguous");
    fs::create_dir(&ambiguous).unwrap();
    fs::write(ambiguous.join("qif.json"), "{}").unwrap();
    fs::write(ambiguous.join("manifest.json"), "{}").unwrap();
    assert!(ModelSource::open(&ambiguous).is_err());
    fs::remove_dir_all(root).unwrap();
}
