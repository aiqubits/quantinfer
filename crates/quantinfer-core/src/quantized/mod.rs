//! Logical quantized storage and descriptor primitives.

mod descriptor;
mod k;
mod q4;
mod q8;

pub const Q8_0_BLOCK_SIZE: usize = 32;
/// Logical output-row storage layout used by quantized matrix providers.
pub const OUTPUT_ROWS_LAYOUT: &str = "output_rows";
pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_K_BLOCK_SIZE: usize = 256;

pub use descriptor::{
    QuantizationAxis, QuantizationScheme, QuantizedTensorDescriptor, ScaleType, StorageType,
    TensorStorageDescriptor, ZeroPointType,
};
pub use k::{Q5KBlock, Q6KBlock, QuantizedTensorQ5K, QuantizedTensorQ6K};
pub use q4::{Q4KBlock, Q4_0Block, QuantizedTensor4, QuantizedTensorQ4K};
pub use q8::{Q8_0Block, QuantizedTensor};
