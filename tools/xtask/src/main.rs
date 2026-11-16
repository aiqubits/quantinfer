//! Repository validation and benchmark task runner.

mod bench;
mod conformance;
mod qif_bench;
mod tokenizer_golden;

use std::env;

fn main() {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("bench") => bench::run(),
        Some("conformance") => conformance::run(),
        Some("qif-bench") => qif_bench::run(),
        Some("tokenizer-golden") => tokenizer_golden::run(),
        None | Some("--help") | Some("-h") => print_help(),
        Some(command) => {
            eprintln!("quantinfer-xtask: unknown command {command}");
            std::process::exit(1);
        }
    }
}

pub(crate) fn option_value(flag: &str) -> Option<String> {
    env::args()
        .skip(2)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn print_help() {
    println!("Usage: cargo run -p quantinfer-xtask -- bench|qif-bench [--qif PATH] [--safetensors PATH] [--gguf PATH] [--tensor NAME]|conformance|tokenizer-golden --tokenizer PATH --golden PATH");
}
