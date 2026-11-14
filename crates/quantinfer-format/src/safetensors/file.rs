//! In-memory and file-backed SafeTensors readers.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use super::decode::{decode_tensor_f32, decode_tensor_u32, decode_tensor_u64};
use super::json::parse_header;
use super::types::{SafeTensorInfo, MAX_HEADER_BYTES};
use crate::{read_file_nofollow, FormatError, Result};

pub(crate) fn open_readonly_nofollow(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    options.open(path)
}

#[derive(Debug, Clone)]
pub struct SafeTensorFile {
    data: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, SafeTensorInfo>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensorFile {
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::open(
            read_file_nofollow(path).map_err(|error| {
                FormatError::Io(format!("cannot read {}: {error}", path.display()))
            })?,
        )
    }

    pub fn open(data: Vec<u8>) -> Result<Self> {
        if data.len() < 8 {
            return Err(FormatError::Invalid(
                "SafeTensors file is truncated".to_owned(),
            ));
        }
        let header_size =
            usize::try_from(u64::from_le_bytes(data[0..8].try_into().map_err(|_| {
                FormatError::Invalid("invalid SafeTensors header".to_owned())
            })?))
            .map_err(|_| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        if header_size > MAX_HEADER_BYTES {
            return Err(FormatError::Invalid(
                "SafeTensors header exceeds the size limit".to_owned(),
            ));
        }
        let header_start: usize = 8;
        let data_start = header_start
            .checked_add(header_size)
            .ok_or_else(|| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        let header_end = data_start;
        let header = data
            .get(header_start..header_end)
            .ok_or_else(|| FormatError::Invalid("SafeTensors header is truncated".to_owned()))?;
        let (tensors, metadata) = parse_header(header, data_start, data.len())?;
        Ok(Self {
            data,
            data_start,
            tensors,
            metadata,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, SafeTensorInfo> {
        &self.tensors
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let start = self
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors start offset overflows usize".to_owned())
            })?;
        let end = self
            .data_start
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors end offset overflows usize".to_owned())
            })?;
        self.data
            .get(start..end)
            .ok_or_else(|| FormatError::Invalid("SafeTensors data is truncated".to_owned()))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        decode_tensor_f32(info, self.tensor_bytes(name)?, name)
    }

    pub fn tensor_u32(&self, name: &str) -> Result<Vec<u32>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        decode_tensor_u32(info, self.tensor_bytes(name)?, name)
    }

    pub fn tensor_u64(&self, name: &str) -> Result<Vec<u64>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        decode_tensor_u64(info, self.tensor_bytes(name)?, name)
    }
}

/// File-backed SafeTensors reader. Tensor reads seek to the requested payload.
#[derive(Debug, Clone)]
pub struct SafeTensorFileReader {
    path: PathBuf,
    data_start: usize,
    tensors: BTreeMap<String, SafeTensorInfo>,
}

impl SafeTensorFileReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = open_readonly_nofollow(&path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        let file_len = usize::try_from(
            file.metadata()
                .map_err(|error| {
                    FormatError::Io(format!("cannot stat {}: {error}", path.display()))
                })?
                .len(),
        )
        .map_err(|_| FormatError::Invalid("SafeTensors file size overflows usize".to_owned()))?;
        let mut size_bytes = [0_u8; 8];
        file.read_exact(&mut size_bytes).map_err(|error| {
            FormatError::Io(format!(
                "cannot read SafeTensors header {}: {error}",
                path.display()
            ))
        })?;
        let header_size = usize::try_from(u64::from_le_bytes(size_bytes))
            .map_err(|_| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        if header_size > MAX_HEADER_BYTES {
            return Err(FormatError::Invalid(
                "SafeTensors header exceeds the size limit".to_owned(),
            ));
        }
        let data_start = 8usize
            .checked_add(header_size)
            .ok_or_else(|| FormatError::Invalid("SafeTensors header overflows usize".to_owned()))?;
        if data_start > file_len {
            return Err(FormatError::Invalid(
                "SafeTensors header is truncated".to_owned(),
            ));
        }
        let mut header = vec![0_u8; header_size];
        file.read_exact(&mut header).map_err(|error| {
            FormatError::Io(format!(
                "cannot read SafeTensors header {}: {error}",
                path.display()
            ))
        })?;
        let (tensors, _) = parse_header(&header, data_start, file_len)?;
        Ok(Self {
            path,
            data_start,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, SafeTensorInfo> {
        &self.tensors
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let path = &self.path;
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let byte_start = self
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors start offset overflows usize".to_owned())
            })?;
        let byte_length = info.data_offsets.1 - info.data_offsets.0;
        let mut file = open_readonly_nofollow(path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        file.seek(SeekFrom::Start(u64::try_from(byte_start).map_err(
            |_| FormatError::Invalid(format!("tensor {name} offset overflows u64")),
        )?))
        .map_err(|error| FormatError::Io(format!("cannot seek {}: {error}", path.display())))?;
        let mut bytes = vec![0_u8; byte_length];
        file.read_exact(&mut bytes).map_err(|error| {
            FormatError::Io(format!(
                "cannot read tensor {name} from {}: {error}",
                path.display()
            ))
        })?;
        decode_tensor_f32(info, &bytes, name)
    }

    pub fn tensor_u32(&self, name: &str) -> Result<Vec<u32>> {
        let path = &self.path;
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let byte_start = self
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors start offset overflows usize".to_owned())
            })?;
        let byte_length = info.data_offsets.1 - info.data_offsets.0;
        let mut file = open_readonly_nofollow(path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        file.seek(SeekFrom::Start(u64::try_from(byte_start).map_err(
            |_| FormatError::Invalid(format!("tensor {name} offset overflows u64")),
        )?))
        .map_err(|error| FormatError::Io(format!("cannot seek {}: {error}", path.display())))?;
        let mut bytes = vec![0_u8; byte_length];
        file.read_exact(&mut bytes).map_err(|error| {
            FormatError::Io(format!(
                "cannot read tensor {name} from {}: {error}",
                path.display()
            ))
        })?;
        decode_tensor_u32(info, &bytes, name)
    }

    pub fn tensor_u64(&self, name: &str) -> Result<Vec<u64>> {
        let path = &self.path;
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))?;
        let byte_start = self
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors start offset overflows usize".to_owned())
            })?;
        let byte_length = info.data_offsets.1 - info.data_offsets.0;
        let mut file = open_readonly_nofollow(path)
            .map_err(|error| FormatError::Io(format!("cannot read {}: {error}", path.display())))?;
        file.seek(SeekFrom::Start(u64::try_from(byte_start).map_err(
            |_| FormatError::Invalid(format!("tensor {name} offset overflows u64")),
        )?))
        .map_err(|error| FormatError::Io(format!("cannot seek {}: {error}", path.display())))?;
        let mut bytes = vec![0_u8; byte_length];
        file.read_exact(&mut bytes).map_err(|error| {
            FormatError::Io(format!(
                "cannot read tensor {name} from {}: {error}",
                path.display()
            ))
        })?;
        decode_tensor_u64(info, &bytes, name)
    }
}
