//! Streaming GGUF payload spool.

use std::io::{self, Read, Seek, Write};

use super::types::{FormatError, GgufWriter};
use super::writer::GgufHeader;
use super::{Result, DEFAULT_ALIGNMENT};

/// Writes tensor payloads to a scratch stream while retaining only the GGUF header.
pub struct GgufSpool<F> {
    scratch: F,
    header: Option<GgufHeader>,
    writer: GgufWriter,
}

impl<F: Read + Seek + Write> GgufSpool<F> {
    /// The scratch stream must be empty and positioned at its beginning.
    pub fn new(scratch: F, writer: GgufWriter) -> Self {
        Self {
            scratch,
            header: Some(GgufHeader::new(writer.alignment)),
            writer,
        }
    }

    /// Encode a tensor with the usual writer and immediately spool its payload.
    pub fn append(&mut self, encode: impl FnOnce(&mut GgufWriter) -> Result<()>) -> Result<()> {
        let mut header = self.header.take().ok_or_else(|| {
            FormatError::Invalid("cannot continue a failed GGUF stream".to_owned())
        })?;
        encode(&mut self.writer)?;
        for tensor in self.writer.tensors.drain(..) {
            let padding = header.append(&tensor)?;
            self.scratch
                .write_all(&[0; DEFAULT_ALIGNMENT as usize][..padding])
                .and_then(|()| self.scratch.write_all(&tensor.data))
                .map_err(|error| FormatError::Io(format!("cannot spool GGUF tensor: {error}")))?;
        }
        self.header = Some(header);
        Ok(())
    }

    pub fn finish(mut self, output: &mut impl Write) -> Result<()> {
        self.append(|_| Ok(()))?;
        let header = self.header.take().expect("append validated the header");
        let data_size = header.data_size;
        let bytes = header.encode(self.writer.metadata)?;
        let copied = self
            .scratch
            .rewind()
            .and_then(|()| output.write_all(&bytes))
            .and_then(|()| io::copy(&mut self.scratch.take(data_size), output))
            .map_err(|error| FormatError::Io(format!("cannot write GGUF: {error}")))?;
        if copied != data_size {
            return Err(FormatError::Io(
                "GGUF scratch stream was truncated".to_owned(),
            ));
        }
        Ok(())
    }
}
