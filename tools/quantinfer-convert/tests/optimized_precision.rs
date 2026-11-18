use std::{collections::BTreeMap, fs, path::Path, process::Command};

use quantinfer_format::{sha256_hex, GgmlType, PackedTensorReader};
use quantinfer_quant::{dequantize_q4_0, dequantize_q8_0};
use serde_json::{json, Value};

fn fixture(root: &Path) {
    fs::write(root.join("config.json"),r#"{"model_type":"llama","vocab_size":32,"hidden_size":32,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"max_position_embeddings":32,"tie_word_embeddings":true,"architectures":["LlamaForCausalLM"]}"#).unwrap();
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (seed, name) in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    .iter()
    .enumerate()
    {
        let shape = if name.contains("norm") {
            vec![32]
        } else {
            vec![32, 32]
        };
        let start = payload.len();
        for index in 0..shape.iter().product::<usize>() {
            let value = if shape.len() == 1 {
                1.0
            } else {
                ((index * 37 + seed * 19) % 211) as f32 * 0.00273 - 0.27
            };
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).into(),
            json!({"dtype":"F32","shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    fs::write(
        root.join("weights.safetensors"),
        [
            (header.len() as u64).to_le_bytes().as_slice(),
            &header,
            &payload,
        ]
        .concat(),
    )
    .unwrap();
    let importance = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    .iter()
    .enumerate()
    .map(|(i, name)| {
        (
            format!("blk.0.{name}.weight"),
            (0..32)
                .map(|j| 1.0 + ((i + j) % 7) as f32 * 0.25)
                .collect::<Vec<_>>(),
        )
    })
    .collect::<BTreeMap<_, _>>();
    fs::write(
        root.join("importance.json"),
        serde_json::to_vec(&importance).unwrap(),
    )
    .unwrap();
}

fn command(root: &Path, output: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
    cmd.arg("--config")
        .arg(root.join("config.json"))
        .arg("--weights")
        .arg(root.join("weights.safetensors"))
        .arg("--output")
        .arg(root.join(output));
    cmd
}

fn policy(budget: u64) -> Value {
    json!({"maximum_encoded_weight_bytes":budget,"candidates":["q4_0","q8_0"],"importance_weights":"importance.json"})
}

fn write_policy(root: &Path, value: &Value) {
    fs::write(root.join("policy.json"), serde_json::to_vec(value).unwrap()).unwrap();
}

#[test]
fn global_precision_budget_matches_actual_gguf_and_both_qif_profiles() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    fixture(root);
    let result = command(root, "f32.gguf")
        .args(["--output-format", "gguf"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let baseline = PackedTensorReader::open_path(root.join("f32.gguf")).unwrap();
    // F32 embedding + three norms, seven Q4_0 projections, and exactly two Q8 upgrades.
    write_policy(root, &policy(4480 + 7 * 576 + 2 * 512));
    let importance: BTreeMap<String, Vec<f32>> =
        serde_json::from_slice(&fs::read(root.join("importance.json")).unwrap()).unwrap();
    let mut expected_payloads = BTreeMap::new();
    for profile in ["gguf", "single", "package"] {
        let name = format!("optimized-{profile}");
        let report_name = format!("report-{profile}.json");
        let mut cmd = command(root, &name);
        cmd.arg("--mixed-precision-optimize")
            .arg(root.join("policy.json"))
            .arg("--mixed-precision-report")
            .arg(root.join(&report_name));
        if profile == "gguf" {
            cmd.args(["--output-format", "gguf"])
                .arg("--manifest")
                .arg(root.join("manifest.json"));
        } else {
            cmd.args(["--qif-profile", profile, "--qif-codec", "zstd"]);
        }
        let result = cmd.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let artifact = PackedTensorReader::open_path(root.join(&name)).unwrap();
        if profile == "gguf" {
            let manifest: Value =
                serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest["quantization"], "Mixed");
        }
        if artifact.is_file_backed() {
            artifact.verify_qif().unwrap();
        }
        let report: Value =
            serde_json::from_slice(&fs::read(root.join(report_name)).unwrap()).unwrap();
        assert_eq!(report["selected_encoded_weight_bytes"], 9536);
        assert_eq!(report["variant"], "mixed");
        assert_eq!(
            artifact
                .tensors()
                .iter()
                .map(|t| t.size as u64)
                .sum::<u64>(),
            9536
        );
        let mut total = 0.0;
        let mut q8_count = 0;
        for tensor in report["tensors"].as_array().unwrap() {
            let name = tensor["name"].as_str().unwrap();
            let reference = baseline.tensor_f32(name).unwrap();
            let restored = match artifact.tensor_info(name).unwrap().dtype {
                GgmlType::Q8_0 => {
                    q8_count += 1;
                    dequantize_q8_0(&artifact.tensor_q8_0(name).unwrap()).unwrap()
                }
                GgmlType::Q4_0 => dequantize_q4_0(&artifact.tensor_q4_0(name).unwrap()).unwrap(),
                other => panic!("unexpected selected type {other:?}"),
            };
            let error = reference
                .as_f32()
                .unwrap()
                .iter()
                .zip(restored.as_f32().unwrap())
                .enumerate()
                .map(|(i, (&a, &b))| {
                    (f64::from(a) - f64::from(b)).powi(2) * f64::from(importance[name][i / 32])
                })
                .sum::<f64>();
            assert!(
                (error - tensor["selected"]["weighted_sse"].as_f64().unwrap()).abs()
                    <= error * 1.0e-12 + 1.0e-20
            );
            assert_eq!(
                tensor["input_f32_sha256"],
                sha256_hex(baseline.tensor_data(name).unwrap().as_ref())
            );
            total += error;
        }
        assert_eq!(q8_count, 2);
        assert!((total - report["total_weighted_sse"].as_f64().unwrap()).abs() < total * 1.0e-12);
        for tensor in artifact.tensors() {
            let payload = sha256_hex(artifact.tensor_data(&tensor.name).unwrap().as_ref());
            if profile == "gguf" {
                expected_payloads.insert(tensor.name.clone(), payload);
            } else {
                assert_eq!(expected_payloads[&tensor.name], payload);
            }
        }
        assert!(!report["artifact_sha256"].as_object().unwrap().is_empty());
    }
}

#[test]
fn rejects_infeasible_or_conflicting_policies_without_publishing_outputs() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    fixture(root);
    for (index, value, extra, expected) in [
        (0, policy(1), vec![], "budget"),
        (
            1,
            json!({"maximum_encoded_weight_bytes":10000,"candidates":["q4_0"],"typo":1}),
            vec![],
            "unknown field",
        ),
        (
            2,
            policy(10000),
            vec!["--quantize", "q8_0"],
            "do not combine",
        ),
        (
            3,
            json!({"maximum_encoded_weight_bytes":10000,"candidates":["q4_0","q8_0"],"max_frontier_states":1}),
            vec![],
            "state limit",
        ),
    ] {
        write_policy(root, &value);
        let output = format!("rejected-{index}");
        let report = root.join(format!("rejected-{index}.json"));
        let result = command(root, &output)
            .arg("--mixed-precision-optimize")
            .arg(root.join("policy.json"))
            .arg("--mixed-precision-report")
            .arg(&report)
            .args(extra)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!root.join(output).exists());
        assert!(!report.exists());
    }
    write_policy(root, &policy(10000));
    for report in [root.join("same"), root.join("policy.json")] {
        let result = command(root, "same")
            .arg("--mixed-precision-optimize")
            .arg(root.join("policy.json"))
            .arg("--mixed-precision-report")
            .arg(report)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!root.join("same").exists());
    }
}
