//! quantinfer-xtask executable.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::quantize_q8_0;

fn main() {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("bench") => run_bench(),
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
    let backend = CpuScalar::new();

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
}

fn print_help() {
    println!("Usage: cargo run -p quantinfer-xtask -- bench");
}
