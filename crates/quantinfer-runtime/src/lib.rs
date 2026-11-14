//! Runtime state shared by model execution and generation.

#![forbid(unsafe_code)]

mod attention;
mod batch;
mod cache;
mod calibration;
mod error;
mod generation;
mod ggml;
mod linear;
pub mod llama;
mod lora;
mod model;
pub mod profile;
pub mod provider;
pub mod qwen3;
pub mod scheduler;
pub mod session;
mod storage;

pub use batch::DecodeBatchItem;
pub use cache::KvCache;
pub use calibration::ProjectionObserver;
pub use error::{Result, RuntimeError, StreamControl, TokenSink};
pub(crate) use generation::last_logits;
pub use generation::{
    argmax, generate_qwen3_stream, generate_stream, select_token, GenerationConfig,
};
pub use linear::LinearWeight;
pub use llama::{LlamaLayerWeights, LlamaModel};
pub use lora::{LoraAdapter, LoraProjection};
pub use model::DecoderModel;
pub(crate) use model::{lm_head_source, tensor_source};
pub use profile::{InferenceProfile, MemoryPlan};
pub use provider::*;
pub use qwen3::{Qwen3LayerWeights, Qwen3Model};
pub use scheduler::{DecodeEvent, DecodeScheduler};
pub use session::{DecodeSession, DecodeStopReason};
pub use storage::{
    canonical_dense_tensor_name, dense_storage_name, is_matrix_storage_name, transpose_dense_layout,
};

#[cfg(test)]
mod tests;
