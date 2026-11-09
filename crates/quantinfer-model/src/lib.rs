//! Model configuration and architecture definitions.

#![forbid(unsafe_code)]

pub mod llama;
pub mod qwen3;
pub mod semantic;

pub use llama::LlamaConfig;
pub use qwen3::Qwen3Config;
pub use semantic::*;
