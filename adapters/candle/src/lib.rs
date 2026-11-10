//! Optional interoperability between QuantInfer and Candle tensors.

#![forbid(unsafe_code)]

#[cfg(feature = "candle")]
mod interop;

#[cfg(feature = "candle")]
pub use interop::*;
