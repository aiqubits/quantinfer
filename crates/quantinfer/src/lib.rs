//! Public QuantInfer facade.

#![forbid(unsafe_code)]

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "candle")]
pub use quantinfer_adapter_candle as candle;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use quantinfer_backend_wgpu as wgpu;
pub use quantinfer_core as core;
pub use quantinfer_format as format;
pub use quantinfer_kernel as kernel;
pub use quantinfer_model as model;
pub use quantinfer_quant as quant;
pub use quantinfer_runtime as runtime;
pub use quantinfer_tokenizer as tokenizer;
