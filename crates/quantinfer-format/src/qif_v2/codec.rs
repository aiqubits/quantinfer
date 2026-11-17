use std::io::{Cursor, Read};

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
        CODEC_NONE => {
            if bytes.len() != expected_length {
                return Err(invalid("uncompressed chunk length mismatch"));
            }
            bytes.to_vec()
        }
        CODEC_ZSTD => {
            let mut decoder = zstd::stream::read::Decoder::new(Cursor::new(bytes))
                .map_err(|error| invalid(format!("zstd decompression failed: {error}")))?;
            // Bound decoder history as well as output. Our writer's level-3
            // frames fit this window; larger windows require a future profile.
            decoder
                .window_log_max(27)
                .map_err(|error| invalid(format!("zstd window limit failed: {error}")))?;
            let limit = u64::try_from(expected_length)
                .ok()
                .and_then(|length| length.checked_add(1))
                .ok_or_else(|| invalid("decoded chunk length overflows"))?;
            let mut output = Vec::new();
            decoder
                .take(limit)
                .read_to_end(&mut output)
                .map_err(|error| invalid(format!("zstd decompression failed: {error}")))?;
            output
        }
        CODEC_LZ4 => {
            let prefix = bytes
                .get(..4)
                .ok_or_else(|| invalid("truncated lz4 size prefix"))?;
            let length = u32::from_le_bytes(prefix.try_into().unwrap()) as usize;
            if length != expected_length {
                return Err(invalid("lz4 size prefix disagrees with chunk length"));
            }
            let mut output = Vec::new();
            output
                .try_reserve_exact(length)
                .map_err(|error| invalid(format!("cannot allocate lz4 output: {error}")))?;
            output.resize(length, 0);
            let decoded = lz4_flex::decompress_into(&bytes[4..], &mut output)
                .map_err(|error| invalid(format!("lz4 decompression failed: {error}")))?;
            if decoded != length {
                return Err(invalid("lz4 decoded chunk length mismatch"));
            }
            output
        }
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
