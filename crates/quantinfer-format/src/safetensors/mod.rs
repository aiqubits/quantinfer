//! SafeTensors readers and shard validation.

mod decode;
mod file;
mod json;
mod shards;
mod types;

pub use file::{SafeTensorFile, SafeTensorFileReader};
pub use shards::SafeTensorShards;
pub use types::{SafeTensorDType, SafeTensorInfo};

#[cfg(test)]
mod tests;
