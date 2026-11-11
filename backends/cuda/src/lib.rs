//! Optional CUDA compute for QuantInfer.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(feature = "native", any(target_os = "linux", target_os = "windows")))]
mod native;

#[cfg(all(feature = "native", any(target_os = "linux", target_os = "windows")))]
pub use native::CudaBackend;
