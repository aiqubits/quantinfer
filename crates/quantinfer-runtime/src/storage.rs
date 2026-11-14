//! Dense model storage names at the GGUF runtime boundary.
//!
//! The semantic role-to-storage mapping belongs at this composition boundary,
//! while source-name canonicalization and matrix transforms are shared by the
//! format crate.

pub use quantinfer_format::{
    canonical_dense_tensor_name, is_matrix_storage_name, transpose_dense_layout,
};
use quantinfer_model::TensorRole;

pub fn dense_storage_name(role: TensorRole, layer: Option<usize>) -> String {
    let suffix = match role {
        TensorRole::TokenEmbedding => "token_embd",
        TensorRole::FinalNormalization => "output_norm",
        TensorRole::LmHead => "output",
        TensorRole::InputNormalization => "attn_norm",
        TensorRole::QueryProjection => "attn_q",
        TensorRole::QueryNormalization => "attn_q_norm",
        TensorRole::KeyProjection => "attn_k",
        TensorRole::KeyNormalization => "attn_k_norm",
        TensorRole::ValueProjection => "attn_v",
        TensorRole::OutputProjection => "attn_output",
        TensorRole::PostAttentionNormalization => "ffn_norm",
        TensorRole::GateProjection => "ffn_gate",
        TensorRole::UpProjection => "ffn_up",
        TensorRole::DownProjection => "ffn_down",
    };
    match layer {
        Some(layer) => format!("blk.{layer}.{suffix}.weight"),
        None => format!("{suffix}.weight"),
    }
}
