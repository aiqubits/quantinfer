//! Optional WASI-NN host integration for native QuantInfer models.

#![forbid(unsafe_code)]

#[cfg(feature = "wasmtime-host")]
mod host;

#[cfg(feature = "wasmtime-host")]
pub use host::{QuantInferBackend, WasiNnLimits};
