//! Streaming GGUF payload spool.

use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::reader::{
    gguf_uses_ggml_rows, gguf_uses_legacy_blocks, GgufControls, PackedTensorReader,
};
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

    pub fn finish(self, output: &mut impl Write) -> Result<()> {
        let (mut scratch, bytes, data_size) = self.finish_parts()?;
        let copied = scratch
            .rewind()
            .and_then(|()| output.write_all(&bytes))
            .and_then(|()| io::copy(&mut scratch.take(data_size), output))
            .map_err(|error| FormatError::Io(format!("cannot write GGUF: {error}")))?;
        if copied != data_size {
            return Err(FormatError::Io(
                "GGUF scratch stream was truncated".to_owned(),
            ));
        }
        Ok(())
    }

    /// Transcode the spool's GGUF tensor bytes to SINGLE using a scratch stream
    /// for the encoded payload. Both streams are rewound independently by the writer.
    pub fn finish_qif(
        self,
        model_id: impl Into<String>,
        output: &mut impl Write,
        payload_scratch: &mut (impl Read + Write + Seek),
        codec: crate::qif::Codec,
    ) -> Result<()> {
        crate::qif_v2::write_spooled_single(
            self.into_source()?,
            model_id,
            output,
            payload_scratch,
            codec,
        )
    }

    /// Transcode the spool and stream shard files directly to a verified package.
    pub fn finish_qif_package(
        self,
        model_id: impl Into<String>,
        output: impl AsRef<Path>,
        max_shard_bytes: Option<usize>,
        codec: crate::qif::Codec,
    ) -> Result<()> {
        crate::qif_v2::write_spooled_package(
            self.into_source()?,
            model_id,
            output.as_ref(),
            max_shard_bytes,
            codec,
        )
    }

    fn finish_parts(mut self) -> Result<(F, Vec<u8>, u64)> {
        self.append(|_| Ok(()))?;
        let header = self.header.take().expect("append validated the header");
        let data_size = header.data_size;
        let bytes = header.encode(self.writer.metadata)?;
        Ok((self.scratch, bytes, data_size))
    }

    fn into_source(self) -> Result<SpoolSource<F>> {
        let (scratch, bytes, payload_size) = self.finish_parts()?;
        let length = usize::try_from(payload_size)
            .ok()
            .and_then(|size| bytes.len().checked_add(size))
            .ok_or_else(|| FormatError::Invalid("GGUF spool length overflows usize".to_owned()))?;
        let controls = PackedTensorReader::parse_gguf_controls(&bytes, length)?;
        let names = controls
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        Ok(SpoolSource {
            scratch,
            controls,
            names,
        })
    }
}

pub(crate) struct SpoolSource<F> {
    scratch: F,
    pub controls: GgufControls,
    names: BTreeMap<String, usize>,
}

impl<F: Read + Seek> SpoolSource<F> {
    pub fn ggml_rows(&self) -> Result<bool> {
        gguf_uses_ggml_rows(&self.controls.metadata, &self.controls.tensors)
    }

    pub fn legacy_blocks(&self) -> Result<bool> {
        gguf_uses_legacy_blocks(&self.controls.metadata)
    }

    pub fn read_tensor(&mut self, name: &str) -> Result<(super::GgmlType, Vec<u8>)> {
        let index = self
            .names
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown spooled tensor {name}")))?;
        let tensor = &self.controls.tensors[*index];
        let mut data = Vec::new();
        data.try_reserve_exact(tensor.size).map_err(|error| {
            FormatError::Invalid(format!("cannot allocate spooled tensor {name}: {error}"))
        })?;
        data.resize(tensor.size, 0);
        self.scratch
            .seek(SeekFrom::Start(tensor.offset))
            .and_then(|_| self.scratch.read_exact(&mut data))
            .map_err(|error| {
                FormatError::Io(format!("cannot read spooled tensor {name}: {error}"))
            })?;
        Ok((tensor.dtype, data))
    }
}
