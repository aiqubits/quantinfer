//! GGML storage decoding and physical matrix layout helpers.

mod layout;
mod loader;

pub(crate) use layout::{load_ggml_linear, load_ggml_matrix};
pub(crate) use loader::load_ggml_as_float;
