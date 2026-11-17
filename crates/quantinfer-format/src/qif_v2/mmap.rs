#![allow(unsafe_code)]

use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::{FormatError, Result};

use super::Reader;

/// Read-only memory-mapped QIF-SINGLE file.
///
/// The unsafe operating-system mapping is isolated in this module. Callers
/// receive the same validated reader as the in-memory API and cannot access an
/// unvalidated byte range through this type. File immutability is a separate
/// caller obligation; a read-only mapping alone does not establish it.
pub struct MappedFile {
    mapping: Mmap,
}

impl MappedFile {
    /// Map a file whose contents will remain immutable for this mapping's lifetime.
    ///
    /// # Safety
    ///
    /// The backing file must not be modified or truncated, by any thread or
    /// process, until this mapping and every borrowed reader/view are dropped.
    /// Read-only permissions, an open file handle and format/hash validation
    /// alone do not provide this guarantee. Use `FileReader` when the caller
    /// cannot uphold it.
    ///
    /// ```compile_fail,E0133
    /// use quantinfer_format::qif::MappedFile;
    /// // File-backed mmap requires an explicit immutability safety argument.
    /// let _ = MappedFile::open("model.qif");
    /// ```
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|error| FormatError::Io(format!("cannot open QIF for mmap: {error}")))?;
        let length = file
            .metadata()
            .map_err(|error| FormatError::Io(format!("cannot stat QIF for mmap: {error}")))?
            .len();
        if length == 0 {
            return Err(FormatError::Invalid(
                "cannot mmap an empty QIF file".to_owned(),
            ));
        }
        // SAFETY: the caller guarantees the backing file remains immutable
        // for the mapping's lifetime, including all borrowed readers/views.
        // Mmap owns the OS mapping after this File handle is closed.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| FormatError::Io(format!("cannot mmap QIF: {error}")))?;
        Ok(Self { mapping })
    }

    pub fn reader(&self) -> Result<Reader<'_>> {
        Reader::open(&self.mapping)
    }

    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
}
