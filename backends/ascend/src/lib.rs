//! Optional Ascend CANN runtime probe.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "native")]
mod native;

#[cfg(feature = "native")]
pub use native::{
    AscendBackend, AscendMatmulExecutor, AscendProvider, AscendRuntime, AscendRuntimeError,
};
