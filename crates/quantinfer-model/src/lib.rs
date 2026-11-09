//! Model configuration and architecture definitions.

#![forbid(unsafe_code)]

pub mod qwen3;
pub mod semantic;

pub use qwen3::Qwen3Config;
pub use semantic::*;
