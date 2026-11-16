#![allow(unsafe_code)]

use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::{FormatError, Result};

use super::QifV2Reader;

/// Read-only memory-mapped QIF2-SINGLE file.
///
/// The unsafe operating-system mapping is isolated in this module. Callers
/// receive the same validated reader as the in-memory API and cannot access an
/// unvalidated byte range through this type.
pub struct QifV2MappedFile {
    mapping: Mmap,
}

impl QifV2MappedFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|error| FormatError::Io(format!("cannot open QIF v2 for mmap: {error}")))?;
        let length = file
            .metadata()
            .map_err(|error| FormatError::Io(format!("cannot stat QIF v2 for mmap: {error}")))?
            .len();
        if length == 0 {
            return Err(FormatError::Invalid(
                "cannot mmap an empty QIF v2 file".to_owned(),
            ));
        }
        // SAFETY: the mapping is read-only, the File is kept open until the
        // mapping has been created, and Mmap owns the resulting mapping.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| FormatError::Io(format!("cannot mmap QIF v2: {error}")))?;
        Ok(Self { mapping })
    }

    pub fn reader(&self) -> Result<QifV2Reader<'_>> {
        QifV2Reader::open(&self.mapping)
    }

    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
}
