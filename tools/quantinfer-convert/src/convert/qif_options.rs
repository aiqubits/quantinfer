//! Shared QIF publication policy for GGUF and SafeTensors conversion.
use crate::common::optional_value;
use quantinfer_format::{
    qif::{import_gguf_package, import_gguf_with_codec, Codec},
    GgufReader,
};
use std::path::Path;

pub(super) struct QifOptions {
    package: bool,
    codec: Codec,
    shard_bytes: Option<usize>,
}

impl QifOptions {
    pub(super) fn parse(arguments: &[String]) -> Result<Self, String> {
        let package = match optional_value(arguments, "--qif-profile").as_deref() {
            None | Some("single") => false,
            Some("package") => true,
            _ => return Err("--qif-profile must be single or package".into()),
        };
        let codec = match optional_value(arguments, "--qif-codec").as_deref() {
            None | Some("none") => Codec::None,
            Some("zstd") => Codec::Zstd,
            Some("lz4") => Codec::Lz4,
            _ => return Err("--qif-codec must be none, zstd or lz4".into()),
        };
        let shard_bytes = optional_value(arguments, "--max-shard-bytes")
            .map(|value| {
                value
                    .parse::<usize>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "--max-shard-bytes must be a positive integer".to_owned())
            })
            .transpose()?;
        if shard_bytes.is_some() && !package {
            return Err("--max-shard-bytes requires --qif-profile package".into());
        }
        Ok(Self {
            package,
            codec,
            shard_bytes,
        })
    }

    pub(super) fn publish(
        &self,
        reader: &GgufReader,
        model_id: &str,
        output: &Path,
    ) -> Result<(), String> {
        if self.package {
            return import_gguf_package(reader, model_id, output, self.shard_bytes, self.codec)
                .map_err(|error| error.to_string());
        }
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary =
            tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
        import_gguf_with_codec(reader, model_id, temporary.as_file_mut(), self.codec)
            .map_err(|error| error.to_string())?;
        temporary
            .persist_noclobber(output)
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}
