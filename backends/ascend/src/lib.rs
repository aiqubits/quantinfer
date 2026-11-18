//! Optional Ascend CANN execution and runtime interfaces.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "native")]
mod native;

#[cfg(feature = "native")]
pub use native::{
    AscendBackend, AscendMatmulExecutor, AscendMatmulPlan, AscendProvider, AscendRuntime,
    AscendRuntimeError,
};

#[cfg(feature = "native")]
pub use native::CannMatmulExecutor;
