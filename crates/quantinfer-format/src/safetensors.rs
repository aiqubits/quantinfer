use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use super::{sha256_reader, FormatError, Result};
use half::{bf16, f16};

const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    I64,
    U64,
}

impl SafeTensorDType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I32" => Ok(Self::I32),
            "U32" => Ok(Self::U32),
            "I64" => Ok(Self::I64),
            "U64" => Ok(Self::U64),
            _ => Err(FormatError::Invalid(format!(
                "unsupported SafeTensors dtype {value}"
            ))),
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
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
            super::read_file_nofollow(path).map_err(|error| {
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

fn open_readonly_nofollow(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    options.open(path)
}

#[derive(Debug, Clone)]
/// Index of file-backed SafeTensors shards. Keep the source files available and
/// unchanged while reading tensors from the index.
pub struct SafeTensorShards {
    index_path: PathBuf,
    tensor_to_shard: BTreeMap<String, String>,
    shards: BTreeMap<String, SafeTensorFileReader>,
}

impl SafeTensorShards {
    pub fn open(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref().to_path_buf();
        let index_bytes = super::read_file_nofollow(&index_path).map_err(|error| {
            FormatError::Io(format!("cannot read {}: {error}", index_path.display()))
        })?;
        let value: serde_json::Value = serde_json::from_slice(&index_bytes).map_err(|error| {
            FormatError::Invalid(format!(
                "invalid SafeTensors index {}: {error}",
                index_path.display()
            ))
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                FormatError::Invalid("SafeTensors index is missing weight_map".to_owned())
            })?;
        let mut tensor_to_shard = BTreeMap::new();
        for (tensor, shard) in weight_map {
            let shard = shard.as_str().ok_or_else(|| {
                FormatError::Invalid(format!("SafeTensors shard for {tensor} is not a string"))
            })?;
            tensor_to_shard.insert(tensor.clone(), shard.to_owned());
        }
        let parent = index_path.parent().unwrap_or_else(|| Path::new("."));
        let mut shards = BTreeMap::new();
        for shard in tensor_to_shard.values() {
            if shards.contains_key(shard) {
                continue;
            }
            let path = resolve_shard_path(parent, shard)?;
            let file = SafeTensorFileReader::open(&path).map_err(|error| {
                FormatError::Io(format!(
                    "cannot open SafeTensors shard {}: {error}",
                    path.display()
                ))
            })?;
            shards.insert(shard.clone(), file);
        }
        for (tensor, shard) in &tensor_to_shard {
            let file = shards.get(shard).ok_or_else(|| {
                FormatError::Invalid(format!("missing loaded SafeTensors shard {shard}"))
            })?;
            if !file.tensors.contains_key(tensor) {
                return Err(FormatError::Invalid(format!(
                    "tensor {tensor} is not present in indexed shard {shard}"
                )));
            }
        }
        Ok(Self {
            index_path,
            tensor_to_shard,
            shards,
        })
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &SafeTensorInfo)> {
        self.tensor_to_shard.iter().filter_map(|(name, shard)| {
            self.shards
                .get(shard)
                .and_then(|file| file.tensors.get(name))
                .map(|info| (name.as_str(), info))
        })
    }

    pub fn tensor_shard(&self, name: &str) -> Result<&str> {
        self.tensor_to_shard
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| FormatError::Invalid(format!("unknown SafeTensors tensor {name}")))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_f32(name)
    }

    pub fn tensor_u32(&self, name: &str) -> Result<Vec<u32>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_u32(name)
    }

    pub fn tensor_u64(&self, name: &str) -> Result<Vec<u64>> {
        let shard = self.tensor_shard(name)?;
        self.shards[shard].tensor_u64(name)
    }

    pub fn verify_shard_checksum(&self, shard: &str, expected_sha256: &str) -> Result<()> {
        let parent = self.index_path.parent().unwrap_or_else(|| Path::new("."));
        if !self.shards.contains_key(shard) {
            return Err(FormatError::Invalid(format!(
                "unknown SafeTensors shard {shard}"
            )));
        }
        let path = resolve_shard_path(parent, shard)?;
        let mut file = open_readonly_nofollow(&path)
            .map_err(|error| FormatError::Io(format!("cannot read shard {shard}: {error}")))?;
        let actual = sha256_reader(&mut file)
            .map_err(|error| FormatError::Io(format!("cannot hash shard {shard}: {error}")))?;
        if actual != expected_sha256 {
            return Err(FormatError::Invalid(format!(
                "checksum mismatch for shard {shard}: expected {expected_sha256}, got {actual}"
            )));
        }
        Ok(())
    }

    pub fn verify_checksums(&self, expected: &BTreeMap<String, String>) -> Result<()> {
        for shard in self.shards.keys() {
            let checksum = expected.get(shard).ok_or_else(|| {
                FormatError::Invalid(format!("missing checksum for shard {shard}"))
            })?;
            self.verify_shard_checksum(shard, checksum)?;
        }
        Ok(())
    }
}

fn resolve_shard_path(parent: &Path, shard: &str) -> Result<PathBuf> {
    let relative = Path::new(shard);
    let mut has_normal = false;
    for component in relative.components() {
        match component {
            std::path::Component::Normal(_) => has_normal = true,
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(FormatError::Invalid(format!(
                    "invalid SafeTensors shard path {shard}"
                )))
            }
        }
    }
    if !has_normal {
        return Err(FormatError::Invalid(format!(
            "invalid SafeTensors shard path {shard}"
        )));
    }
    let parent = fs::canonicalize(parent).map_err(|error| {
        FormatError::Io(format!(
            "cannot resolve SafeTensors shard directory: {error}"
        ))
    })?;
    let path = fs::canonicalize(parent.join(relative)).map_err(|error| {
        FormatError::Io(format!("cannot resolve SafeTensors shard {shard}: {error}"))
    })?;
    if !path.starts_with(&parent) {
        return Err(FormatError::Invalid(format!(
            "SafeTensors shard {shard} escapes its index directory"
        )));
    }
    Ok(path)
}

fn decode_tensor_f32(info: &SafeTensorInfo, bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    let values: Vec<f32> = match info.dtype {
        SafeTensorDType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        SafeTensorDType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
            .collect(),
        SafeTensorDType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
            .collect(),
        SafeTensorDType::I32
        | SafeTensorDType::U32
        | SafeTensorDType::I64
        | SafeTensorDType::U64 => {
            return Err(FormatError::Invalid(format!(
                "tensor {name} has integer dtype {:?}, expected a floating dtype",
                info.dtype
            )))
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} contains non-finite values"
        )));
    }
    Ok(values)
}

fn decode_tensor_u32(info: &SafeTensorInfo, bytes: &[u8], name: &str) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    match info.dtype {
        SafeTensorDType::I32 | SafeTensorDType::U32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()),
        dtype => Err(FormatError::Invalid(format!(
            "tensor {name} has dtype {dtype:?}, expected I32 or U32"
        ))),
    }
}

fn decode_tensor_u64(info: &SafeTensorInfo, bytes: &[u8], name: &str) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(info.dtype.element_size()) {
        return Err(FormatError::Invalid(format!(
            "tensor {name} has a non-aligned byte length"
        )));
    }
    match info.dtype {
        SafeTensorDType::I64 | SafeTensorDType::U64 => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| u64::from_le_bytes(*chunk))
            .collect()),
        dtype => Err(FormatError::Invalid(format!(
            "tensor {name} has dtype {dtype:?}, expected I64 or U64"
        ))),
    }
}

fn parse_header(
    header: &[u8],
    data_start: usize,
    file_len: usize,
) -> Result<(BTreeMap<String, SafeTensorInfo>, BTreeMap<String, String>)> {
    let value: serde_json::Value = serde_json::from_slice(header)
        .map_err(|error| FormatError::Invalid(format!("invalid SafeTensors JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| FormatError::Invalid("SafeTensors header must be an object".to_owned()))?;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, entry) in object {
        if name == "__metadata__" {
            if let Some(entries) = entry.as_object() {
                for (key, value) in entries {
                    if let Some(value) = value.as_str() {
                        metadata.insert(key.clone(), value.to_owned());
                    }
                }
            }
            continue;
        }
        let entry = entry
            .as_object()
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} is not an object")))?;
        let dtype = SafeTensorDType::parse(
            entry
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no dtype")))?,
        )?;
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no shape")))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| {
                        FormatError::Invalid(format!("tensor {name} has an invalid shape"))
                    })
                    .and_then(|value| {
                        usize::try_from(value).map_err(|_| {
                            FormatError::Invalid(format!("tensor {name} shape overflows usize"))
                        })
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no data offsets")))?;
        if offsets.len() != 2 {
            return Err(FormatError::Invalid(format!(
                "tensor {name} must have two data offsets"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has invalid offsets")))
            .and_then(|value| {
                usize::try_from(value).map_err(|_| {
                    FormatError::Invalid(format!("tensor {name} start offset overflows usize"))
                })
            })?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has invalid offsets")))
            .and_then(|value| {
                usize::try_from(value).map_err(|_| {
                    FormatError::Invalid(format!("tensor {name} end offset overflows usize"))
                })
            })?;
        if start > end {
            return Err(FormatError::Invalid(format!(
                "tensor {name} offsets exceed file bounds"
            )));
        }
        let byte_start = data_start.checked_add(start).ok_or_else(|| {
            FormatError::Invalid(format!("tensor {name} start exceeds file bounds"))
        })?;
        let byte_end = data_start.checked_add(end).ok_or_else(|| {
            FormatError::Invalid(format!("tensor {name} end exceeds file bounds"))
        })?;
        if byte_end > file_len {
            return Err(FormatError::Invalid(format!(
                "tensor {name} offsets exceed file bounds"
            )));
        }
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} shape overflows usize")))
        })?;
        let expected_size = elements
            .checked_mul(dtype.element_size())
            .ok_or_else(|| FormatError::Invalid(format!("tensor {name} size overflows usize")))?;
        if byte_end - byte_start != expected_size {
            return Err(FormatError::Invalid(format!(
                "tensor {name} byte length does not match its shape"
            )));
        }
        tensors.insert(
            name.clone(),
            SafeTensorInfo {
                dtype,
                shape,
                data_offsets: (start, end),
            },
        );
    }
    Ok((tensors, metadata))
}

#[cfg(test)]
mod tests {
    use super::{SafeTensorDType, SafeTensorFile, SafeTensorShards};

    #[test]
    fn reads_f32_tensor_and_metadata() {
        let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        let file = SafeTensorFile::open(bytes).unwrap();
        assert_eq!(file.metadata().get("format"), Some(&"pt".to_owned()));
        assert_eq!(file.tensor_f32("weight").unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn rejects_non_finite_floating_tensors() {
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        let file = SafeTensorFile::open(bytes).unwrap();
        assert!(file.tensor_f32("weight").is_err());
    }

    #[test]
    fn reads_signed_and_unsigned_packed_integer_tensors() {
        let header = br#"{"signed":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"unsigned":{"dtype":"U32","shape":[1],"data_offsets":[4,8]},"signed64":{"dtype":"I64","shape":[1],"data_offsets":[8,16]},"unsigned64":{"dtype":"U64","shape":[1],"data_offsets":[16,24]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&0x1234_5678_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&0x1234_5678_9abc_def0_u64.to_le_bytes());
        let file = SafeTensorFile::open(bytes).unwrap();
        assert_eq!(file.tensors()["signed"].dtype, SafeTensorDType::I32);
        assert_eq!(file.tensor_u32("signed").unwrap(), vec![u32::MAX]);
        assert_eq!(file.tensor_u32("unsigned").unwrap(), vec![0x1234_5678]);
        assert_eq!(file.tensor_u64("signed64").unwrap(), vec![u64::MAX]);
        assert_eq!(
            file.tensor_u64("unsigned64").unwrap(),
            vec![0x1234_5678_9abc_def0]
        );
        assert!(file.tensor_f32("signed").is_err());
    }

    #[test]
    fn rejects_tensor_bytes_that_do_not_match_shape() {
        let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        assert!(SafeTensorFile::open(bytes).is_err());
    }

    #[test]
    fn rejects_oversized_headers_before_allocating_them() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&((64 * 1024 * 1024 + 1) as u64).to_le_bytes());
        assert!(SafeTensorFile::open(bytes).is_err());
    }

    #[test]
    fn loads_tensors_from_a_sharded_index_without_merging_files() {
        let directory =
            std::env::temp_dir().join(format!("quantinfer-shards-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let write_shard = |name: &str, tensor: &str, value: f32| {
            let header =
                format!(r#"{{"{tensor}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
            std::fs::write(directory.join(name), bytes).unwrap();
        };
        write_shard("model-00001-of-00002.safetensors", "a", 1.0);
        write_shard("model-00002-of-00002.safetensors", "b", 2.0);
        let index_path = directory.join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            r#"{"metadata":{"total_size":8},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let shards = SafeTensorShards::open(&index_path).unwrap();
        assert_eq!(
            shards.tensor_shard("a").unwrap(),
            "model-00001-of-00002.safetensors"
        );
        assert_eq!(shards.tensor_f32("b").unwrap(), vec![2.0]);
        assert_eq!(shards.tensors().count(), 2);
        let second = directory.join("model-00002-of-00002.safetensors");
        let checksum = crate::sha256_hex(&std::fs::read(&second).unwrap());
        shards
            .verify_shard_checksum("model-00002-of-00002.safetensors", &checksum)
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&second)
            .unwrap()
            .set_len(std::fs::metadata(&second).unwrap().len() - 1)
            .unwrap();
        let error = shards.tensor_f32("b").unwrap_err().to_string();
        assert!(error.contains("model-00002-of-00002.safetensors"));
        assert_eq!(shards.tensor_f32("a").unwrap(), vec![1.0]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_shard_paths_outside_the_index_directory() {
        let directory =
            std::env::temp_dir().join(format!("quantinfer-shard-path-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let index_path = directory.join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
        )
        .unwrap();
        let error = SafeTensorShards::open(&index_path).unwrap_err().to_string();
        assert!(error.contains("invalid SafeTensors shard path"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
