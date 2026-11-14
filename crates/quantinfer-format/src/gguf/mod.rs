//! Compatibility format readers and writers.

mod encoding;
mod layout;
mod reader;
mod types;
mod writer;

pub type Result<T> = std::result::Result<T, types::FormatError>;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
pub const GGML_OUTPUT_ROWS_LAYOUT: &str = "ggml_output_rows";
const DEFAULT_ALIGNMENT: u64 = 32;
pub(crate) const QUANT_BLOCK_LAYOUT: &str = "quantinfer.ggml.block_layout";
const MAX_GGUF_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_GGUF_ARRAY_ITEMS: usize = 1_048_576;
const MAX_GGUF_TENSOR_RANK: usize = 1024;

pub(crate) use encoding::{encode_value, tensor_byte_size, write_string, Cursor};
pub use layout::qk_channel;
pub use reader::{GgufReader, PackedTensorReader};
pub use types::{FormatError, GgmlType, GgufTensorInfo, GgufValue, GgufWriter};
pub use writer::GgufSpool;

#[cfg(test)]
mod tests;
