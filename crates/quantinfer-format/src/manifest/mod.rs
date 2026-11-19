//! Model manifests, checksums, and verified packages.

mod hash;
pub(crate) use hash::hex_encode;
pub(crate) use hash::open_readonly_nofollow;
mod package;
mod types;

pub use hash::{read_file_nofollow, sha256_hex, sha256_reader};
pub use package::ModelPackage;
pub use types::{
    Compatibility, ExecutionVariant, KernelRequirements, ModelManifest, TensorManifest,
};

#[cfg(test)]
mod tests;
