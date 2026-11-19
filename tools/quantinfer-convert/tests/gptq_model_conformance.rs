#![cfg(feature = "gptq")]

use quantinfer::{
    cpu::{CpuModel, ModelSource},
    runtime::{generate_stream, DecoderModel, GenerationConfig, StreamControl},
};
use quantinfer_format::{qif, sha256_hex, GgmlType};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command, time::Instant};

const SOURCE_SHA: &str = "2549efafb6629d3b44a3e1eba18bc2a222fcca24302fc2b7abe3f8b2d17cc762";
const CALIBRATION: &str = include_str!("../../../tests/fixtures/gptq/calibration.jsonl");
const VALIDATION: &str = include_str!("../../../tests/fixtures/gptq/validation.jsonl");

fn run(model: &CpuModel, tokens: &[usize]) -> Value {
    let mut cache = model.new_cache(64).unwrap();
    let logits = model.forward(tokens, &mut cache).unwrap();
    let raw: Vec<u8> = logits
        .as_f32()
        .unwrap()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let generated = generate_stream(
        model,
        tokens,
        &mut model.new_cache(64).unwrap(),
        &GenerationConfig {
            max_new_tokens: 8,
            eos_token_id: Some(0),
            ..Default::default()
        },
        &mut |_| Ok(StreamControl::Continue),
    )
    .unwrap();
    // Verify the actual generating cache too, rather than the separate prefill.
    let mut generated_cache = model.new_cache(64).unwrap();
    let repeated = generate_stream(
        model,
        tokens,
        &mut generated_cache,
        &GenerationConfig {
            max_new_tokens: 8,
            eos_token_id: Some(0),
            ..Default::default()
        },
        &mut |_| Ok(StreamControl::Continue),
    )
    .unwrap();
    assert_eq!(generated, repeated);
    let expected = tokens.len() + generated.len() - usize::from(generated.last() == Some(&0));
    for layer in 0..30 {
        assert_eq!(generated_cache.sequence_length(layer).unwrap(), expected);
    }
    json!({"full_logits_sha256":sha256_hex(&raw),"generated":generated,"kv_length":expected})
}

#[test]
#[ignore = "requires the pinned SmolLM2 F32 GGUF; performs all 210 projection quantizations"]
fn whole_model_gptq_meets_quality_and_lossless_publication_gates() {
    let input = PathBuf::from(
        std::env::var_os("QUANTINFER_GPTQ_WORKFLOW_SOURCE").expect("verified source"),
    );
    let report_path =
        PathBuf::from(std::env::var_os("QUANTINFER_GPTQ_WORKFLOW_REPORT").expect("report output"));
    let golden: Value =
        serde_json::from_str(include_str!("../../../tests/golden/smollm2-135m-f32.json")).unwrap();
    let tokens: Vec<usize> = golden["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let source_hashes = {
        let bytes = fs::read(&input).unwrap();
        assert_eq!(sha256_hex(&bytes), SOURCE_SHA);
        let reader = quantinfer_format::GgufReader::open(bytes).unwrap();
        let model = CpuModel::from_gguf(&reader, None, None).unwrap();
        let logits = model
            .forward(&tokens, &mut model.new_cache(64).unwrap())
            .unwrap();
        let last = &logits.as_f32().unwrap()[(tokens.len() - 1) * model.vocab_size()..];
        for sample in golden["logits"].as_array().unwrap() {
            assert!(
                (f64::from(last[sample[0].as_u64().unwrap() as usize])
                    - sample[1].as_f64().unwrap())
                .abs()
                    < 0.001
            );
        }
        reader
            .tensors()
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    sha256_hex(reader.tensor_bytes(&t.name).unwrap()),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    let root = tempfile::tempdir().unwrap();
    let dataset = root.path().join("calibration.jsonl");
    let validation = root.path().join("validation.jsonl");
    fs::write(&dataset, CALIBRATION).unwrap();
    fs::write(&validation, VALIDATION).unwrap();
    let output = root.path().join("model.qif");
    let quality = root.path().join("quality.json");
    let start = Instant::now();
    let process = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"))
        .arg("gptq")
        .arg("--input")
        .arg(&input)
        .arg("--dataset")
        .arg(&dataset)
        .arg("--validation-dataset")
        .arg(&validation)
        .arg("--output")
        .arg(&output)
        .arg("--report")
        .arg(&quality)
        .args([
            "--max-sequence-length",
            "64",
            "--maximum-kl",
            "0.5",
            "--minimum-top10-overlap",
            "6",
        ])
        .output()
        .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    fs::write(
        report_path.with_extension("workflow.log"),
        [process.stdout.as_slice(), process.stderr.as_slice()].concat(),
    )
    .unwrap();
    assert!(
        process.status.success(),
        "{}",
        String::from_utf8_lossy(&process.stderr)
    );
    let quality: Value = serde_json::from_slice(&fs::read(&quality).unwrap()).unwrap();
    assert_eq!(quality["quality_status"], "passed");
    assert_eq!(quality["quantized_projections"], 210);
    assert_eq!(quality["calibration_prompts"], 13);
    assert_eq!(quality["calibration_token_rows"], 159);
    assert_eq!(
        quality["calibration_dataset_sha256"],
        sha256_hex(CALIBRATION.as_bytes())
    );
    assert_eq!(
        quality["validation_dataset_sha256"],
        sha256_hex(VALIDATION.as_bytes())
    );
    assert_eq!(quality["holdouts"].as_array().unwrap().len(), 4);
    for sample in quality["holdouts"].as_array().unwrap() {
        assert!(sample["kl"].as_f64().unwrap() <= 0.5);
        assert!(sample["top10_overlap"].as_u64().unwrap() >= 6);
    }
    let reader = ModelSource::open(&output)
        .unwrap()
        .open_reader(None)
        .unwrap();
    reader.verify_qif().unwrap();
    let mut hashes = BTreeMap::new();
    for tensor in reader.tensors() {
        let hash = sha256_hex(reader.tensor_data(&tensor.name).unwrap().as_ref());
        if quantinfer_format::is_linear_projection_storage_name(&tensor.name) {
            assert_eq!(tensor.dtype, GgmlType::Q4_0);
            let record = quality["projections"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v["tensor"] == tensor.name)
                .unwrap();
            assert_eq!(record["payload_sha256"], hash);
            assert_eq!(record["calibration_rows"], 159);
        } else {
            assert_eq!(hash, source_hashes[&tensor.name]);
        }
        hashes.insert(tensor.name.clone(), hash);
    }
    let model = CpuModel::from_gguf(&reader, None, None).unwrap();
    let single = run(&model, &tokens);
    drop(model);
    let package = root.path().join("package");
    qif::write_qif_package_with_codec(
        reader.embedded_manifest().unwrap().clone(),
        reader.qif_metadata().unwrap(),
        |n| reader.tensor_bytes(n),
        &package,
        Some(16 * 1024 * 1024),
        qif::Codec::Zstd,
    )
    .unwrap();
    let packaged = ModelSource::open(&package)
        .unwrap()
        .open_reader(None)
        .unwrap();
    packaged.verify_qif().unwrap();
    for tensor in packaged.tensors() {
        assert_eq!(
            sha256_hex(packaged.tensor_data(&tensor.name).unwrap().as_ref()),
            hashes[&tensor.name]
        );
    }
    let model = CpuModel::from_gguf(&packaged, None, None).unwrap();
    let actual = run(&model, &tokens);
    assert_eq!(actual, single);
    assert_eq!(
        quantinfer_format::sha256_reader(&mut fs::File::open(&input).unwrap()).unwrap(),
        SOURCE_SHA
    );
    let report = json!({"source_sha256":SOURCE_SHA,"whole_model_seconds":seconds,"quality":quality,"tensor_raw_sha256":hashes,
        "single":single,"package":actual,"package_index_sha256":sha256_hex(&fs::read(package.join("index.qifi")).unwrap()),"passed":true});
    fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!(
        "gptq_whole_model_quality={}",
        json!({"seconds":seconds,"holdouts":report["quality"]["holdouts"],"quantized_projections":210,"profiles_equal":true})
    );
}
