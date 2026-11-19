#![cfg(feature = "cpu")]

use quantinfer::{
    cpu::{CpuModel, ModelSource},
    format::{
        qif::{self, Codec, Reader},
        sha256_hex, GgmlType, GgufReader,
    },
    runtime::{generate_stream, DecoderModel, GenerationConfig, StreamControl},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct OwnedDirectory(PathBuf);
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires a verified native-layout SmolLM2 F32 GGUF"]
fn real_mixed_layout_weights_preserve_smollm2_logits_generation_and_storage() {
    let source_path = PathBuf::from(
        std::env::var_os("QUANTINFER_LAYOUT_SOURCE_GGUF").expect("verified source path"),
    );
    let source_bytes = fs::read(&source_path).unwrap();
    let source_hash = sha256_hex(&source_bytes);
    assert_eq!(
        source_hash,
        std::env::var("QUANTINFER_LAYOUT_SOURCE_SHA256").expect("verified source SHA256")
    );
    let source = GgufReader::open(source_bytes).unwrap();
    assert!(!source.uses_ggml_output_rows().unwrap());
    assert!(source.tensors().iter().all(|t| t.dtype == GgmlType::F32));
    let golden: Value =
        serde_json::from_str(include_str!("../../../tests/golden/smollm2-135m-f32.json")).unwrap();
    let prompt: Vec<usize> = golden["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let tokens: Vec<usize> = golden["generated_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let config = GenerationConfig {
        max_new_tokens: tokens.len(),
        eos_token_id: Some(0),
        ..GenerationConfig::default()
    };
    let reference = CpuModel::from_gguf(&source, None, None).unwrap();
    let spec = CpuModel::spec_from_gguf(&source, None).unwrap();
    assert_eq!(spec.hidden_size, 576);
    assert_eq!(spec.layers.len(), 30);
    let expected = reference
        .forward(&prompt, &mut reference.new_cache(64).unwrap())
        .unwrap();
    assert!(expected.as_f32().unwrap().iter().all(|v| v.is_finite()));
    let vocabulary = spec.vocabulary.size;
    let last = &expected.as_f32().unwrap()[(prompt.len() - 1) * vocabulary..];
    let sampled_reference_error = golden["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let index = pair[0].as_u64().unwrap() as usize;
            (f64::from(last[index]) - pair[1].as_f64().unwrap()).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        sampled_reference_error <= 0.001,
        "{sampled_reference_error}"
    );
    let baseline_generated = generate_stream(
        &reference,
        &prompt,
        &mut reference.new_cache(64).unwrap(),
        &config,
        &mut |_| Ok(StreamControl::Continue),
    )
    .unwrap();
    assert_eq!(baseline_generated, tokens);
    drop(reference);

    let root = std::env::temp_dir().join(format!(
        "quantinfer-real-layout-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let owned = OwnedDirectory(root);
    let mut intermediate = Vec::new();
    qif::import_gguf(&source, "smollm2-layout-conformance", &mut intermediate).unwrap();
    let native = Reader::open(&intermediate).unwrap();
    let mut manifest = native.manifest().clone();
    let mut changed = BTreeMap::new();
    for name in ["blk.0.attn_k.weight", "blk.0.ffn_up.weight"] {
        let tensor = manifest
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .unwrap();
        let shape = tensor.descriptor.logical_shape.dims();
        assert_ne!(shape[0], shape[1]);
        let original = source.tensor_f32(name).unwrap();
        let raw = original.as_f32().unwrap();
        let mut physical = vec![0.0; raw.len()];
        for output in 0..shape[1] {
            for input in 0..shape[0] {
                physical[output * shape[0] + input] = raw[input * shape[1] + output];
            }
        }
        changed.insert(
            name.to_owned(),
            physical
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        tensor.descriptor.physical_layout = "ggml_output_rows".into();
        tensor.checksum_sha256 = None;
    }
    let metadata = native.metadata().clone();
    let get = |name: &str| {
        if let Some(raw) = changed.get(name) {
            Ok(raw.as_slice())
        } else {
            native.tensor_bytes(name)
        }
    };
    let single = owned.0.join("model.qif");
    qif::write_qif(
        manifest.clone(),
        &metadata,
        get,
        &mut fs::File::create(&single).unwrap(),
    )
    .unwrap();
    let package = owned.0.join("package");
    qif::write_qif_package_with_codec(
        manifest,
        &metadata,
        get,
        &package,
        Some(64 * 1024 * 1024),
        Codec::Zstd,
    )
    .unwrap();
    drop(native);
    drop(intermediate);
    let mut results = Vec::new();
    for (profile, path) in [("single/none", &single), ("package/zstd", &package)] {
        let reader = ModelSource::open(path).unwrap().open_reader(None).unwrap();
        assert!(reader.is_file_backed());
        assert!(!reader.uses_ggml_output_rows().unwrap());
        reader.verify_qif().unwrap();
        let mut inventory = BTreeMap::new();
        for info in reader.tensors() {
            let bytes = reader.tensor_data(&info.name).unwrap();
            let expected_bytes = if let Some(raw) = changed.get(&info.name) {
                raw.as_slice()
            } else {
                source.tensor_bytes(&info.name).unwrap()
            };
            assert_eq!(bytes.as_ref(), expected_bytes, "{}", info.name);
            assert_eq!(info.shape, source.tensor_info(&info.name).unwrap().shape);
            assert_eq!(
                reader.tensor_storage_layout(&info.name).unwrap(),
                if changed.contains_key(&info.name) {
                    "ggml_output_rows"
                } else {
                    "row_major"
                }
            );
            inventory.insert(info.name.clone(), sha256_hex(bytes.as_ref()));
        }
        let model = CpuModel::from_gguf(&reader, None, None).unwrap();
        let actual = model
            .forward(&prompt, &mut model.new_cache(64).unwrap())
            .unwrap();
        assert_eq!(actual, expected);
        let mut emitted = Vec::new();
        let mut cache = model.new_cache(64).unwrap();
        let generated = generate_stream(&model, &prompt, &mut cache, &config, &mut |token| {
            emitted.push(token);
            Ok(StreamControl::Continue)
        })
        .unwrap();
        assert_eq!(generated, baseline_generated);
        assert_eq!(emitted, generated);
        let cached = prompt.len() + generated.len()
            - usize::from(generated.last().copied() == config.eos_token_id);
        for layer in 0..spec.layers.len() {
            assert_eq!(cache.sequence_length(layer).unwrap(), cached);
        }
        assert_eq!(reader.cached_tensor_bytes(), 0);
        results.push(json!({"profile":profile,"tensor_count":inventory.len(),"raw_tensor_sha256":inventory,"generated":generated,"kv_length":cached,"full_logits_equal":true}));
    }
    let report = json!({"source_sha256":source_hash,"sampled_transformers_max_error":sampled_reference_error,"changed_layout_tensors":changed.keys().collect::<Vec<_>>(),"profiles":results,"passed":true});
    println!("qif_layout_quality={report}");
    if let Some(path) = std::env::var_os("QUANTINFER_LAYOUT_REPORT") {
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
