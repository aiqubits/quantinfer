//! Native WebGPU compute for QuantInfer.

#![forbid(unsafe_code)]

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
mod native;

#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use native::WgpuBackend;
