//! Resource bounds and scratch space shared by the QIF fuzz targets.

use quantinfer_format::qif::{FileReader, MappedFile, Reader};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

pub const MAX_INPUT: usize = 1024 * 1024;
const MAX_OUTPUT: u64 = 4 * 1024 * 1024;

pub fn workspace() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("quantinfer-fuzz-{}-{now}", std::process::id()));
        fs::create_dir(&root).unwrap();
        root
    })
}

pub fn bounded_lengths(mut lengths: impl Iterator<Item = u64>) -> bool {
    lengths
        .try_fold(0u64, |sum, length| sum.checked_add(length))
        .is_some_and(|sum| sum <= MAX_OUTPUT)
}

pub fn exercise_single(data: &[u8]) {
    let path = workspace().join("single.qif");
    fs::write(&path, data).unwrap();
    let memory = Reader::open(data);
    let file = FileReader::open(&path);
    assert_eq!(memory.is_ok(), file.is_ok());
    if let Ok(reader) = memory {
        // SAFETY: this process exclusively owns the scratch file. The mapping
        // is dropped before the next fuzz input rewrites it.
        let mapped = unsafe { MappedFile::open(&path) }.unwrap();
        assert!(mapped.reader().is_ok());
        if !bounded_lengths(reader.tensors().iter().map(|tensor| tensor.raw_length)) {
            return;
        }
        let mut file = file.unwrap();
        assert_eq!(reader.verify().is_ok(), file.verify().is_ok());
        for tensor in reader.tensors().iter().take(8) {
            let end = tensor.raw_length.min(4096) as usize;
            let memory_range = reader.read_range(&tensor.name, 0..end);
            let file_range = file.read_range(&tensor.name, 0..end);
            assert_eq!(memory_range.is_ok(), file_range.is_ok());
            if let (Ok(lhs), Ok(rhs)) = (memory_range, file_range) {
                assert_eq!(lhs, rhs);
            }
            let _ = reader.tensor_data(&tensor.name);
        }
    }
}

/// Keep the original input for hash/signature rejection, and also repair the
/// unsigned control hash so mutations reach the index/metadata parsers.
pub fn repair_single_root(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 144 || data.len() > MAX_INPUT {
        return None;
    }
    let mut hash = Sha256::new();
    for offset in [24, 40, 56] {
        let start = usize::try_from(u64::from_le_bytes(
            data[offset..offset + 8].try_into().ok()?,
        ))
        .ok()?;
        let length = usize::try_from(u64::from_le_bytes(
            data[offset + 8..offset + 16].try_into().ok()?,
        ))
        .ok()?;
        hash.update(data.get(start..start.checked_add(length)?)?);
    }
    let mut bytes = data.to_vec();
    bytes[88..120].copy_from_slice(&hash.finalize());
    Some(bytes)
}
