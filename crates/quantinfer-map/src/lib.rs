//! Read-only mapping of a model file.
//!
//! This is the only crate that calls `memmap2`. Mapping is an unsafe OS call
//! because shrinking the file while the map is alive is undefined.

use std::fs::File;
use std::io;
use std::ops::Deref;

/// A read-only map of a file that dereferences to its bytes.
pub struct MappedFile {
    map: memmap2::Mmap,
}

impl MappedFile {
    /// Map `file` read-only.
    ///
    /// The caller must keep the file from shrinking for as long as this value
    /// is alive. The length is the file length observed by the map call.
    pub fn map(file: &File) -> io::Result<Self> {
        // SAFETY: `file` is open. The caller upholds the no-truncate contract
        // documented on this function. The map is read-only.
        let map = unsafe { memmap2::Mmap::map(file)? };
        Ok(Self { map })
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.map
    }
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedFile")
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::MappedFile;

    #[test]
    fn maps_the_bytes_that_were_written() {
        let mut file = tempfile();
        file.write_all(b"qif-bytes").unwrap();
        file.sync_all().unwrap();
        let mapped = MappedFile::map(&file).unwrap();
        assert_eq!(&mapped[..], b"qif-bytes");
    }

    fn tempfile() -> std::fs::File {
        let path = std::env::temp_dir().join(format!(
            "quantinfer-map-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap()
    }
}
