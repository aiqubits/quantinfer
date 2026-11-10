//! quantinfer-xtask executable.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use quantinfer_backend_cpu::{CpuRuntimeConfig, CpuScalar};
use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0};
use quantinfer_tokenizer::{
    ByteTokenizer, ChatMessage, GgufBpeTokenizer, Qwen3ChatTemplate, ThinkingMode, Tokenizer,
};
use serde_json::Value;

fn main() {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("bench") => run_bench(),
        Some("conformance") => run_conformance(),
        Some("tokenizer-golden") => run_tokenizer_golden(),
        None | Some("--help") | Some("-h") => print_help(),
        Some(command) => {
            eprintln!("quantinfer-xtask: unknown command {command}");
            std::process::exit(1);
        }
    }
}

fn run_bench() {
    const ITERATIONS: usize = 32;
    const INPUT_ROWS: usize = 8;
    const INPUT_COLUMNS: usize = 128;
    const OUTPUT_COLUMNS: usize = 64;

    let lhs_values = (0..INPUT_ROWS * INPUT_COLUMNS)
        .map(|index| (index % 17) as f32 / 17.0)
        .collect();
    let rhs_values = (0..INPUT_COLUMNS * OUTPUT_COLUMNS)
        .map(|index| (index % 23) as f32 / 23.0)
        .collect();
    let lhs = Tensor::from_f32(vec![INPUT_ROWS, INPUT_COLUMNS], lhs_values)
        .expect("benchmark lhs shape is valid");
    let rhs = Tensor::from_f32(vec![INPUT_COLUMNS, OUTPUT_COLUMNS], rhs_values)
        .expect("benchmark rhs shape is valid");
    let quantized = quantize_q8_0(&rhs).expect("benchmark rhs is Q8_0 block aligned");
    let quantized_q4 = quantize_q4_0(&rhs).expect("benchmark rhs is Q4_0 block aligned");
    let quantized_q4_k = quantize_q4_k(&rhs).expect("benchmark rhs is Q4_K block aligned");
    let quantized_q5_k = quantize_q5_k(&rhs).expect("benchmark rhs is Q5_K block aligned");
    let quantized_q6_k = quantize_q6_k(&rhs).expect("benchmark rhs is Q6_K block aligned");
    let backend = CpuScalar::with_config(CpuRuntimeConfig::default());
    println!(
        "cpu_isa={:?} threads={} workspace_reuse={}",
        backend.config().isa,
        backend.config().threads,
        backend.config().workspace_reuse
    );

    let started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERATIONS {
        let result = backend
            .matmul_q8_0(black_box(&lhs), black_box(&quantized))
            .expect("benchmark matmul should succeed");
        checksum += result.as_f32().expect("benchmark result is F32")[0];
    }
    let elapsed = started.elapsed();
    println!(
        "q8_0_matmul iterations={ITERATIONS} shape={INPUT_ROWS}x{INPUT_COLUMNS}x{OUTPUT_COLUMNS} elapsed_ns={} checksum={checksum:.6}",
        elapsed.as_nanos()
    );

    let started = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..ITERATIONS {
        let result = backend
            .matmul_q4_0(black_box(&lhs), black_box(&quantized_q4))
            .expect("benchmark matmul should succeed");
        checksum += result.as_f32().expect("benchmark result is F32")[0];
    }
    let elapsed = started.elapsed();
    println!(
        "q4_0_matmul iterations={ITERATIONS} shape={INPUT_ROWS}x{INPUT_COLUMNS}x{OUTPUT_COLUMNS} elapsed_ns={} checksum={checksum:.6}",
        elapsed.as_nanos()
    );

    macro_rules! run_quant_bench {
        ($name:literal, $quantized:expr, $method:ident) => {{
            let started = Instant::now();
            let mut checksum = 0.0;
            for _ in 0..ITERATIONS {
                let result = backend
                    .$method(black_box(&lhs), black_box(&$quantized))
                    .expect("K-quant benchmark matmul should succeed");
                checksum += result.as_f32().expect("benchmark result is F32")[0];
            }
            println!(
                "{}_matmul iterations={} shape={}x{}x{} elapsed_ns={} checksum={:.6}",
                $name,
                ITERATIONS,
                INPUT_ROWS,
                INPUT_COLUMNS,
                OUTPUT_COLUMNS,
                started.elapsed().as_nanos(),
                checksum
            );
        }};
    }
    run_quant_bench!("q4_k", quantized_q4_k, matmul_q4_k);
    run_quant_bench!("q5_k", quantized_q5_k, matmul_q5_k);
    run_quant_bench!("q6_k", quantized_q6_k, matmul_q6_k);
}

fn run_conformance() {
    let tokenizer = ByteTokenizer;
    let prompt = "QuantInfer 中文\n42!";
    let token_ids = tokenizer
        .encode(prompt)
        .expect("fixture tokenizer should encode conformance prompt");
    assert_eq!(tokenizer.decode(&token_ids).unwrap(), prompt);
    let chat_prompt = Qwen3ChatTemplate
        .render(
            &[ChatMessage::new("user", prompt)],
            ThinkingMode::Disabled,
            true,
        )
        .expect("fixture chat template should render");
    assert!(chat_prompt.contains("<|im_start|>assistant"));

    let input = Tensor::from_f32(
        vec![1, 32],
        (0..32).map(|index| index as f32 / 32.0 - 0.5).collect(),
    )
    .expect("fixture input shape is valid");
    let q8 = quantize_q8_0(&input).expect("Q8_0 fixture quantization should succeed");
    let q4 = quantize_q4_0(&input).expect("Q4_0 fixture quantization should succeed");
    let q8_values = quantinfer_quant::dequantize_q8_0(&q8)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let q4_values = quantinfer_quant::dequantize_q4_0(&q4)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let max_error = |values: &[f32]| {
        values
            .iter()
            .zip(input.as_f32().unwrap())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max)
    };
    assert!(max_error(&q8_values) < 0.01);
    assert!(max_error(&q4_values) < 0.08);
    println!(
        "conformance_ok prompt_tokens={} q8_max_error={:.6} q4_max_error={:.6}",
        token_ids.len(),
        max_error(&q8_values),
        max_error(&q4_values)
    );
}

fn run_tokenizer_golden() {
    let tokenizer_path =
        option_value("--tokenizer").expect("tokenizer-golden requires --tokenizer");
    let golden_path = option_value("--golden").expect("tokenizer-golden requires --golden");
    let root: Value =
        serde_json::from_str(&fs::read_to_string(tokenizer_path).expect("read tokenizer.json"))
            .expect("parse tokenizer.json");
    let vocab = root["model"]["vocab"]
        .as_object()
        .expect("tokenizer model.vocab");
    let maximum_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .expect("non-empty vocabulary") as usize;
    let mut tokens = vec![String::new(); maximum_id + 1];
    for (token, id) in vocab {
        tokens[id.as_u64().expect("integer token id") as usize] = token.clone();
    }
    if let Some(added) = root["added_tokens"].as_array() {
        for entry in added {
            if let (Some(id), Some(content)) = (entry["id"].as_u64(), entry["content"].as_str()) {
                let id = id as usize;
                if id >= tokens.len() {
                    tokens.resize(id + 1, String::new());
                }
                tokens[id] = content.to_owned();
            }
        }
    }
    let merges = root["model"]["merges"]
        .as_array()
        .expect("tokenizer model.merges")
        .iter()
        .map(|merge| {
            if let Some(value) = merge.as_str() {
                value.to_owned()
            } else {
                let parts = merge.as_array().expect("merge pair");
                format!(
                    "{} {}",
                    parts[0].as_str().unwrap(),
                    parts[1].as_str().unwrap()
                )
            }
        })
        .collect::<Vec<_>>();
    let tokenizer = GgufBpeTokenizer::new(tokens, &merges, None, None).expect("build tokenizer");
    let golden: Value =
        serde_json::from_str(&fs::read_to_string(golden_path).expect("read golden vectors"))
            .expect("parse golden vectors");
    for vector in golden["vectors"].as_array().expect("golden vectors array") {
        let text = vector["text"].as_str().unwrap();
        let expected = vector["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let actual = tokenizer.encode(text).expect("encode golden text");
        assert_eq!(actual, expected, "tokenizer golden mismatch for {text:?}");
        let decoded = tokenizer.decode(&actual).expect("decode golden ids");
        assert_eq!(
            decoded,
            vector["decoded"].as_str().unwrap(),
            "tokenizer decode mismatch for {text:?}"
        );
    }
    println!(
        "tokenizer_golden_ok vectors={}",
        golden["vectors"].as_array().unwrap().len()
    );
}

fn option_value(flag: &str) -> Option<String> {
    env::args()
        .skip(2)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn print_help() {
    println!("Usage: cargo run -p quantinfer-xtask -- bench|conformance|tokenizer-golden --tokenizer PATH --golden PATH");
}
