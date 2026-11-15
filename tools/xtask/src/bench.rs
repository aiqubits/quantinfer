//! Synthetic CPU benchmark task.

use std::hint::black_box;
use std::time::Instant;

use quantinfer_backend_cpu::{CpuRuntimeConfig, CpuScalar};
use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0};

pub(crate) fn run() {
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
    println!("rss_kb_before={}", current_rss_kb().unwrap_or(0));

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
    println!("rss_kb_after={}", current_rss_kb().unwrap_or(0));
}

fn current_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.split_whitespace().next()?;
        value.parse().ok()
    })
}
