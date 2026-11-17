//! Shared physical chunk verification for memory, files and package shards.
use super::{codec, invalid, payload::PayloadVerifier, CODEC_NONE};
use crate::{FormatError, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

#[derive(Debug, Clone)]
pub(super) struct Chunk {
    pub(super) shard_id: u32,
    pub(super) codec: u32,
    pub(super) raw_offset: u64,
    pub(super) raw_length: u64,
    pub(super) stored_offset: u64,
    pub(super) stored_length: u64,
    pub(super) checksum_sha256: [u8; 32],
}

impl Chunk {
    pub(super) fn verify_memory(
        &self,
        stored: &[u8],
        payload: &mut PayloadVerifier<'_>,
    ) -> Result<()> {
        if stored.len() as u64 != self.stored_length {
            return Err(invalid("chunk length mismatch"));
        }
        let digest: [u8; 32] = Sha256::digest(stored).into();
        if digest != self.checksum_sha256 {
            return Err(invalid("chunk checksum mismatch"));
        }
        if self.codec == CODEC_NONE {
            payload.update(stored)
        } else {
            let decoded = codec::decompress(
                self.codec,
                stored,
                usize::try_from(self.raw_length)
                    .map_err(|_| invalid("raw chunk length overflows usize"))?,
            )?;
            payload.update(&decoded)
        }
    }

    pub(super) fn verify_file(
        &self,
        file: &mut File,
        base: u64,
        payload: &mut PayloadVerifier<'_>,
    ) -> Result<()> {
        let offset = base
            .checked_add(self.stored_offset)
            .ok_or_else(|| invalid("chunk file offset overflows"))?;
        file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
        if self.codec != CODEC_NONE {
            codec::validate_lengths(self.codec, self.raw_length, self.stored_length)?;
            let length = usize::try_from(self.stored_length)
                .map_err(|_| invalid("stored chunk length overflows usize"))?;
            let mut stored = Vec::new();
            stored
                .try_reserve_exact(length)
                .map_err(|error| invalid(format!("cannot allocate compressed chunk: {error}")))?;
            stored.resize(length, 0);
            file.read_exact(&mut stored).map_err(io_error)?;
            return self.verify_memory(&stored, payload);
        }
        let mut buffer = [0; 64 * 1024];
        let mut remaining = self.stored_length;
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..count]).map_err(io_error)?;
            hash.update(&buffer[..count]);
            payload.update(&buffer[..count])?;
            remaining -= count as u64;
        }
        let digest: [u8; 32] = hash.finalize().into();
        if digest != self.checksum_sha256 {
            return Err(invalid("chunk checksum mismatch"));
        }
        Ok(())
    }
}

fn io_error(error: std::io::Error) -> FormatError {
    FormatError::Io(format!("QIF chunk I/O failed: {error}"))
}
