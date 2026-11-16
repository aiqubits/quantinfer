use std::io::Cursor;

use super::{invalid, CODEC_LZ4, CODEC_NONE, CODEC_ZSTD};
use crate::Result;

pub(crate) fn compress(codec: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    match codec {
        CODEC_NONE => Ok(bytes.to_vec()),
        CODEC_ZSTD => zstd::stream::encode_all(Cursor::new(bytes), 3)
            .map_err(|error| invalid(format!("zstd compression failed: {error}"))),
        CODEC_LZ4 => Ok(lz4_flex::compress_prepend_size(bytes)),
        value => Err(invalid(format!("unsupported codec {value}"))),
    }
}

pub(crate) fn decompress(codec: u32, bytes: &[u8], expected_length: usize) -> Result<Vec<u8>> {
    let output = match codec {
        CODEC_NONE => bytes.to_vec(),
        CODEC_ZSTD => zstd::stream::decode_all(Cursor::new(bytes))
            .map_err(|error| invalid(format!("zstd decompression failed: {error}")))?,
        CODEC_LZ4 => lz4_flex::decompress_size_prepended(bytes)
            .map_err(|error| invalid(format!("lz4 decompression failed: {error}")))?,
        value => return Err(invalid(format!("unsupported codec {value}"))),
    };
    if output.len() != expected_length {
        return Err(invalid(format!(
            "decoded chunk has {} bytes, expected {expected_length}",
            output.len()
        )));
    }
    Ok(output)
}
