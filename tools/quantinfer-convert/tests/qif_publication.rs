use quantinfer_format::{GgufReader, GgufValue, GgufWriter, PackedTensorReader};
use std::{fs, process::Command};

#[test]
fn safetensors_publishes_all_native_profiles_and_codecs_with_identical_tensor_values() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.json");
    fs::write(&config, r#"{"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":16,"tie_word_embeddings":true}"#).unwrap();
    let weights = root.path().join("weights.safetensors");
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape) in [
        ("model.embed_tokens.weight", vec![4, 4]),
        ("model.norm.weight", vec![4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
    ] {
        let start = payload.len();
        for index in 0..shape.iter().product::<usize>() {
            payload.extend_from_slice(&((index + 1) as f32 * 0.01).to_le_bytes());
        }
        header.insert(
            name.into(),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let file = [
        (header.len() as u64).to_le_bytes().as_slice(),
        &header,
        &payload,
    ]
    .concat();
    fs::write(&weights, file).unwrap();
    let command = |output: &std::path::Path| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
        command
            .arg("--config")
            .arg(&config)
            .arg("--weights")
            .arg(&weights)
            .arg("--output")
            .arg(output);
        command
    };
    let baseline = root.path().join("baseline.gguf");
    let output = command(&baseline)
        .args(["--output-format", "gguf"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let baseline = GgufReader::open(fs::read(baseline).unwrap()).unwrap();
    for profile in ["single", "package"] {
        for codec in ["none", "zstd", "lz4"] {
            let path = root.path().join(format!("{profile}-{codec}"));
            let output = command(&path)
                .args(["--qif-profile", profile, "--qif-codec", codec])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let reader = PackedTensorReader::open_path(path).unwrap();
            reader.verify_qif().unwrap();
            assert_eq!(reader.tensors().len(), baseline.tensors().len());
            for tensor in baseline.tensors() {
                assert_eq!(
                    reader.tensor_data(&tensor.name).unwrap().as_ref(),
                    baseline.tensor_bytes(&tensor.name).unwrap()
                );
            }
        }
    }
}

#[test]
fn publishes_both_qif_profiles_all_codecs_without_requantization_or_overwrite() {
    let root = std::env::temp_dir().join(format!("qif-publication-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let input = root.join("input.gguf");
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("fixture".into()));
    writer
        .add_f32_tensor("weight", vec![2], &[1.0, 2.0])
        .unwrap();
    fs::write(&input, writer.finish().unwrap()).unwrap();
    let original = GgufReader::open(fs::read(&input).unwrap()).unwrap();
    for profile in ["single", "package"] {
        for codec in ["none", "zstd", "lz4"] {
            let output = root.join(format!("{profile}-{codec}"));
            let command = || {
                let mut cmd = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
                cmd.arg("from-gguf")
                    .arg("--input")
                    .arg(&input)
                    .arg("--output")
                    .arg(&output)
                    .args(["--qif-profile", profile, "--qif-codec", codec]);
                if profile == "package" {
                    cmd.args(["--max-shard-bytes", "4096"]);
                }
                cmd
            };
            let result = command().output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let published = PackedTensorReader::open_path(&output).unwrap();
            assert_eq!(published.cached_tensor_bytes(), 0);
            published.verify_qif().unwrap();
            assert_eq!(
                published.tensor_data("weight").unwrap().as_ref(),
                original.tensor_bytes("weight").unwrap()
            );
            assert!(!command().output().unwrap().status.success());
            PackedTensorReader::open_path(&output)
                .unwrap()
                .verify_qif()
                .unwrap();
        }
    }
    for options in [
        vec!["--qif-codec", "unknown"],
        vec!["--qif-profile", "unknown"],
        vec!["--max-shard-bytes", "4096"],
        vec!["--qif-profile", "package", "--max-shard-bytes", "0"],
    ] {
        let output = root.join("invalid");
        assert!(!Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"))
            .arg("from-gguf")
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .args(options)
            .output()
            .unwrap()
            .status
            .success());
        assert!(!output.exists());
    }
    fs::remove_dir_all(root).unwrap();
}
