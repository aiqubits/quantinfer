//! Reference CPU kernels implemented without a vendor runtime dependency.

#![forbid(unsafe_code)]

mod backend;
mod config;
mod matmul;

pub use config::{CpuIsa, CpuRuntimeConfig, CpuScalar};

#[cfg(test)]
mod tests;
