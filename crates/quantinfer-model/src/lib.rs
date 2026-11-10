//! Model configuration and architecture definitions.

#![forbid(unsafe_code)]

pub mod adapters;
pub mod llama;
pub mod qwen3;
pub mod semantic;

pub use adapters::{canonical_dense_tensor_name, LlamaAdapter, ModelAdapter, Qwen3Adapter};
pub use llama::LlamaConfig;
pub use qwen3::Qwen3Config;
pub use semantic::*;
