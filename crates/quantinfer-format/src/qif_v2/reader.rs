use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use sha2::{Digest, Sha256};

use quantinfer_core::DType;

use crate::gguf::{GgufTensorInfo, PackedTensorReader};
use crate::{GgmlType, GgufValue, ModelManifest, Result};

use super::metadata::QifV2MetadataValue;
use super::{
    checked_range, invalid, metadata_to_gguf, ALIAS_NONE, ALIGNMENT, CODEC_LZ4, CODEC_NONE,
    CODEC_ZSTD, HASH_SHA256, HASH_SIZE, HEADER_SIZE, INDEX_MAGIC, INDEX_VERSION, MAGIC, MAX_CHUNKS,
    MAX_HEADER_SECTION, MAX_INDEX_BYTES, MAX_RANK, MAX_TENSORS, SIGNATURE_LENGTH, VERSION,
};

const INDEX_HEADER_SIZE: usize = 48;
const TENSOR_RECORD_SIZE: usize = 120;
const CHUNK_RECORD_SIZE: usize = 72;
const KNOWN_FLAGS: u32 = 0;
const MAX_MANIFEST_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_DEPTH: usize = 32;

type ParsedControl = (
    ModelManifest,
    BTreeMap<String, QifV2MetadataValue>,
    Vec<QifV2TensorInfo>,
    Vec<ChunkInfo>,
);

/// A tensor entry from the QIF v2 binary index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QifV2TensorInfo {
    pub tensor_id: u64,
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub physical_encoding: u32,
    pub quantization_scheme: u32,
    pub layout: u32,
    pub codec: u32,
    pub raw_length: u64,
    pub stored_length: u64,
    pub checksum_sha256: [u8; HASH_SIZE],
    pub alias_of: Option<u64>,
    pub(crate) chunk_start: usize,
    pub(crate) chunk_count: usize,
}

#[derive(Debug, Clone)]
struct ChunkInfo {
    shard_id: u32,
    codec: u32,
    raw_offset: u64,
    raw_length: u64,
    stored_offset: u64,
    stored_length: u64,
    checksum_sha256: [u8; HASH_SIZE],
}

/// A fully validated QIF v2 single-file artifact.
pub struct QifV2Reader<'a> {
    bytes: &'a [u8],
    manifest: ModelManifest,
    metadata: BTreeMap<String, QifV2MetadataValue>,
    tensors: Vec<QifV2TensorInfo>,
    chunks: Vec<ChunkInfo>,
    payload_offset: usize,
}

impl<'a> QifV2Reader<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let header = bytes
            .get(..HEADER_SIZE)
            .ok_or_else(|| invalid("file is shorter than the fixed header"))?;
        if &header[..4] != MAGIC {
            return Err(invalid("invalid magic"));
        }
        let version = u32_at(header, 4);
        if version != VERSION {
            return Err(invalid(format!("unsupported version {version}")));
        }
        let flags = u32_at(header, 8);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(invalid(format!("unsupported feature flags 0x{flags:08x}")));
        }
        let header_size = usize::try_from(u32_at(header, 12))
            .map_err(|_| invalid("header size overflows usize"))?;
        if header_size != HEADER_SIZE {
            return Err(invalid(format!(
                "unsupported fixed header size {header_size}"
            )));
        }
        if u32_at(header, 16) != HASH_SHA256 {
            return Err(invalid("unsupported hash algorithm"));
        }
        if u32_at(header, 20) != INDEX_VERSION {
            return Err(invalid("unsupported index version"));
        }

        let manifest_offset = u64_at(header, 24);
        let manifest_length = u64_at(header, 32);
        let metadata_offset = u64_at(header, 40);
        let metadata_length = u64_at(header, 48);
        let index_offset = u64_at(header, 56);
        let index_length = u64_at(header, 64);
        let payload_offset = u64_at(header, 72);
        let payload_length = u64_at(header, 80);
        let signature_offset = u64_at(header, 120);
        let signature_length = u64_at(header, 128);
        let root_hash: [u8; HASH_SIZE] = header[88..120]
            .try_into()
            .map_err(|_| invalid("invalid root hash"))?;
        if u64_at(header, 136) != 0 {
            return Err(invalid("reserved header fields must be zero"));
        }

        let sections = [
            ("manifest", manifest_offset, manifest_length),
            ("metadata", metadata_offset, metadata_length),
            ("index", index_offset, index_length),
            ("payload", payload_offset, payload_length),
        ];
        for (name, offset, length) in sections {
            if offset < HEADER_SIZE as u64 {
                return Err(invalid(format!("{name} overlaps the fixed header")));
            }
            if offset % ALIGNMENT as u64 != 0 {
                return Err(invalid(format!("{name} offset is not aligned")));
            }
            let _ = checked_range(bytes, offset, length)?;
            if name != "payload" && length > MAX_HEADER_SECTION as u64 {
                return Err(invalid(format!("{name} exceeds the size limit")));
            }
        }
        if index_length > MAX_INDEX_BYTES as u64 {
            return Err(invalid("index exceeds the size limit"));
        }
        let mut ranges = sections
            .iter()
            .map(|(name, offset, length)| {
                let end = offset
                    .checked_add(*length)
                    .ok_or_else(|| invalid(format!("{name} range overflows")))?;
                Ok((*name, *offset, end))
            })
            .collect::<Result<Vec<_>>>()?;
        ranges.sort_unstable_by_key(|(_, start, _)| *start);
        if ranges.windows(2).any(|pair| pair[1].1 < pair[0].2) {
            return Err(invalid("sections overlap"));
        }
        let signature = if signature_length == 0 {
            if signature_offset != 0 {
                return Err(invalid("signature offset is nonzero with zero length"));
            }
            None
        } else {
            if signature_length != SIGNATURE_LENGTH as u64
                || signature_offset < HEADER_SIZE as u64
                || !signature_offset.is_multiple_of(ALIGNMENT as u64)
            {
                return Err(invalid("invalid QIF2 signature range"));
            }
            let signature_end = signature_offset
                .checked_add(signature_length)
                .ok_or_else(|| invalid("signature range overflows"))?;
            if sections.iter().any(|(_, start, length)| {
                signature_offset < start.saturating_add(*length) && *start < signature_end
            }) {
                return Err(invalid("signature overlaps another section"));
            }
            Some(checked_range(bytes, signature_offset, signature_length)?)
        };
        let file_end = payload_offset
            .checked_add(payload_length)
            .ok_or_else(|| invalid("payload range overflows"))?;
        if file_end != bytes.len() as u64 {
            return Err(invalid("unexpected bytes outside the payload section"));
        }

        let manifest_bytes = checked_range(bytes, manifest_offset, manifest_length)?;
        let metadata_bytes = checked_range(bytes, metadata_offset, metadata_length)?;
        let index_bytes = checked_range(bytes, index_offset, index_length)?;
        let (manifest, metadata, tensors, chunks) = parse_control(
            manifest_bytes,
            metadata_bytes,
            index_bytes,
            payload_length,
            root_hash,
            signature,
        )?;

        Ok(Self {
            bytes,
            manifest,
            metadata,
            tensors,
            chunks,
            payload_offset: usize::try_from(payload_offset)
                .map_err(|_| invalid("payload offset overflows usize"))?,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn metadata(&self) -> &BTreeMap<String, QifV2MetadataValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[QifV2TensorInfo] {
        &self.tensors
    }

    pub fn tensor_info(&self, name: &str) -> Result<&QifV2TensorInfo> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| invalid(format!("unknown tensor {name}")))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&'a [u8]> {
        let tensor = self.tensor_info(name)?;
        self.tensor_bytes_by_info(tensor)
    }

    pub fn tensor_data(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor_info(name)?;
        self.tensor_data_by_info(tensor)
    }

    pub fn read_range(&self, name: &str, range: Range<usize>) -> Result<Vec<u8>> {
        let mut tensor = self.tensor_info(name)?;
        while let Some(alias) = tensor.alias_of {
            tensor = self
                .tensors
                .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("alias target is out of range"))?;
        }
        let raw_length = usize::try_from(tensor.raw_length)
            .map_err(|_| invalid("tensor length overflows usize"))?;
        if range.start > range.end || range.end > raw_length {
            return Err(invalid("tensor range exceeds the stored tensor"));
        }
        let mut output = vec![0_u8; range.end - range.start];
        for chunk in &self.chunks[tensor.chunk_start..tensor.chunk_start + tensor.chunk_count] {
            let chunk_start = usize::try_from(chunk.raw_offset)
                .map_err(|_| invalid("chunk offset overflows usize"))?;
            let chunk_end = chunk_start
                .checked_add(
                    usize::try_from(chunk.raw_length)
                        .map_err(|_| invalid("chunk length overflows usize"))?,
                )
                .ok_or_else(|| invalid("chunk range overflows usize"))?;
            let copy_start = range.start.max(chunk_start);
            let copy_end = range.end.min(chunk_end);
            if copy_start >= copy_end {
                continue;
            }
            let stored = checked_range(
                self.bytes,
                self.payload_offset as u64 + chunk.stored_offset,
                chunk.stored_length,
            )?;
            let decoded;
            let source = if chunk.codec == CODEC_NONE {
                stored
            } else {
                decoded = super::codec::decompress(
                    chunk.codec,
                    stored,
                    usize::try_from(chunk.raw_length)
                        .map_err(|_| invalid("chunk length overflows usize"))?,
                )?;
                &decoded
            };
            let source_start = copy_start - chunk_start;
            let source_end = copy_end - chunk_start;
            let output_start = copy_start - range.start;
            output[output_start..output_start + source_end - source_start]
                .copy_from_slice(&source[source_start..source_end]);
        }
        Ok(output)
    }

    pub fn verify(&self) -> Result<()> {
        for tensor in &self.tensors {
            let bytes = self.tensor_data_by_info(tensor)?;
            let digest: [u8; HASH_SIZE] = Sha256::digest(&bytes).into();
            if digest != tensor.checksum_sha256 {
                return Err(invalid(format!("tensor {} checksum mismatch", tensor.name)));
            }
            for chunk_index in tensor.chunk_start..tensor.chunk_start + tensor.chunk_count {
                let chunk = &self.chunks[chunk_index];
                let stored = checked_range(
                    self.bytes,
                    self.payload_offset as u64 + chunk.stored_offset,
                    chunk.stored_length,
                )?;
                let digest: [u8; HASH_SIZE] = Sha256::digest(stored).into();
                if digest != chunk.checksum_sha256 {
                    return Err(invalid(format!(
                        "tensor {} chunk checksum mismatch",
                        tensor.name
                    )));
                }
            }
        }
        Ok(())
    }

    fn tensor_bytes_by_info(&self, tensor: &QifV2TensorInfo) -> Result<&'a [u8]> {
        if let Some(target) = tensor.alias_of {
            let target = self
                .tensors
                .get(usize::try_from(target).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("alias target is out of range"))?;
            return self.tensor_bytes_by_info(target);
        }
        if tensor.codec != CODEC_NONE || tensor.chunk_count != 1 {
            return Err(invalid(
                "tensor is compressed, multi-chunk, or empty; use tensor_data",
            ));
        }
        let first = &self.chunks[tensor.chunk_start];
        let start = self
            .payload_offset
            .checked_add(
                usize::try_from(first.stored_offset)
                    .map_err(|_| invalid("chunk offset overflows usize"))?,
            )
            .ok_or_else(|| invalid("chunk offset overflows usize"))?;
        let end = start
            .checked_add(
                usize::try_from(tensor.stored_length)
                    .map_err(|_| invalid("tensor length overflows usize"))?,
            )
            .ok_or_else(|| invalid("tensor end overflows usize"))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| invalid("tensor payload is truncated"))
    }

    fn tensor_data_by_info(&self, tensor: &QifV2TensorInfo) -> Result<Vec<u8>> {
        if let Some(target) = tensor.alias_of {
            let target = self
                .tensors
                .get(usize::try_from(target).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("alias target is out of range"))?;
            return self.tensor_data_by_info(target);
        }
        let mut output = Vec::with_capacity(
            usize::try_from(tensor.raw_length)
                .map_err(|_| invalid("tensor length overflows usize"))?,
        );
        for chunk_index in tensor.chunk_start..tensor.chunk_start + tensor.chunk_count {
            let chunk = &self.chunks[chunk_index];
            let stored = checked_range(
                self.bytes,
                self.payload_offset as u64 + chunk.stored_offset,
                chunk.stored_length,
            )?;
            let decoded = super::codec::decompress(
                chunk.codec,
                stored,
                usize::try_from(chunk.raw_length)
                    .map_err(|_| invalid("chunk length overflows usize"))?,
            )?;
            output.extend_from_slice(&decoded);
        }
        if output.len()
            != usize::try_from(tensor.raw_length)
                .map_err(|_| invalid("tensor length overflows usize"))?
        {
            return Err(invalid("decoded tensor length disagrees with index"));
        }
        Ok(output)
    }
}

/// File-backed QIF v2 reader. It reads only the fixed header, manifest,
/// metadata and index at open time; tensor bytes are fetched on demand.
pub struct QifV2FileReader {
    file: File,
    manifest: ModelManifest,
    metadata: BTreeMap<String, QifV2MetadataValue>,
    tensors: Vec<QifV2TensorInfo>,
    chunks: Vec<ChunkInfo>,
    payload_offset: u64,
}

impl QifV2FileReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .map_err(|error| crate::FormatError::Io(format!("cannot open QIF v2: {error}")))?;
        let file_length = file
            .metadata()
            .map_err(|error| crate::FormatError::Io(format!("cannot stat QIF v2: {error}")))?
            .len();
        let mut header = [0_u8; HEADER_SIZE];
        read_at(&mut file, 0, &mut header)?;
        let header = parse_file_header(&header, file_length)?;
        let manifest = read_section(&mut file, header.manifest_offset, header.manifest_length)?;
        let metadata = read_section(&mut file, header.metadata_offset, header.metadata_length)?;
        let index = read_section(&mut file, header.index_offset, header.index_length)?;
        let signature = if header.signature_length == 0 {
            None
        } else {
            Some(read_section(
                &mut file,
                header.signature_offset,
                header.signature_length,
            )?)
        };
        let (manifest, metadata_values, tensors, chunks) = parse_control(
            &manifest,
            &metadata,
            &index,
            header.payload_length,
            header.root_hash,
            signature.as_deref(),
        )?;
        Ok(Self {
            file,
            manifest,
            metadata: metadata_values,
            tensors,
            chunks,
            payload_offset: header.payload_offset,
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn metadata(&self) -> &BTreeMap<String, QifV2MetadataValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[QifV2TensorInfo] {
        &self.tensors
    }

    pub fn tensor_info(&self, name: &str) -> Result<&QifV2TensorInfo> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| invalid(format!("unknown tensor {name}")))
    }

    pub fn read_range(&mut self, name: &str, range: Range<usize>) -> Result<Vec<u8>> {
        let tensor = self.tensor_info(name)?.clone();
        let tensor = resolve_tensor(&self.tensors, tensor)?;
        let raw_length = usize::try_from(tensor.raw_length)
            .map_err(|_| invalid("tensor length overflows usize"))?;
        if range.start > range.end || range.end > raw_length {
            return Err(invalid("tensor range exceeds the stored tensor"));
        }
        let mut output = vec![0_u8; range.end - range.start];
        for chunk in &self.chunks[tensor.chunk_start..tensor.chunk_start + tensor.chunk_count] {
            let chunk_start = usize::try_from(chunk.raw_offset)
                .map_err(|_| invalid("chunk offset overflows usize"))?;
            let chunk_end = chunk_start
                .checked_add(
                    usize::try_from(chunk.raw_length)
                        .map_err(|_| invalid("chunk length overflows usize"))?,
                )
                .ok_or_else(|| invalid("chunk range overflows usize"))?;
            let copy_start = range.start.max(chunk_start);
            let copy_end = range.end.min(chunk_end);
            if copy_start >= copy_end {
                continue;
            }
            let mut stored = vec![
                0_u8;
                usize::try_from(chunk.stored_length)
                    .map_err(|_| invalid("chunk length overflows usize"))?
            ];
            read_at(
                &mut self.file,
                self.payload_offset
                    .checked_add(chunk.stored_offset)
                    .ok_or_else(|| invalid("file offset overflows"))?,
                &mut stored,
            )?;
            let decoded = super::codec::decompress(
                chunk.codec,
                &stored,
                usize::try_from(chunk.raw_length)
                    .map_err(|_| invalid("chunk length overflows usize"))?,
            )?;
            let source_start = copy_start - chunk_start;
            let source_end = copy_end - chunk_start;
            let output_start = copy_start - range.start;
            output[output_start..output_start + source_end - source_start]
                .copy_from_slice(&decoded[source_start..source_end]);
        }
        Ok(output)
    }

    pub fn verify(&mut self) -> Result<()> {
        for tensor in self.tensors.clone() {
            let tensor = resolve_tensor(&self.tensors, tensor)?;
            let raw_length = usize::try_from(tensor.raw_length)
                .map_err(|_| invalid("tensor length overflows usize"))?;
            let bytes = self.read_range(&tensor.name, 0..raw_length)?;
            let digest: [u8; HASH_SIZE] = Sha256::digest(&bytes).into();
            if digest != tensor.checksum_sha256 {
                return Err(invalid(format!("tensor {} checksum mismatch", tensor.name)));
            }
            for chunk in &self.chunks[tensor.chunk_start..tensor.chunk_start + tensor.chunk_count] {
                let mut stored = vec![
                    0_u8;
                    usize::try_from(chunk.stored_length)
                        .map_err(|_| invalid("chunk length overflows usize"))?
                ];
                read_at(
                    &mut self.file,
                    self.payload_offset
                        .checked_add(chunk.stored_offset)
                        .ok_or_else(|| invalid("file offset overflows"))?,
                    &mut stored,
                )?;
                let digest: [u8; HASH_SIZE] = Sha256::digest(&stored).into();
                if digest != chunk.checksum_sha256 {
                    return Err(invalid("chunk checksum mismatch"));
                }
            }
        }
        Ok(())
    }
}

fn resolve_tensor(
    tensors: &[QifV2TensorInfo],
    mut tensor: QifV2TensorInfo,
) -> Result<QifV2TensorInfo> {
    while let Some(alias) = tensor.alias_of {
        tensor = tensors
            .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
            .ok_or_else(|| invalid("alias target is out of range"))?
            .clone();
    }
    Ok(tensor)
}

#[derive(Clone, Copy)]
struct FileHeader {
    manifest_offset: u64,
    manifest_length: u64,
    metadata_offset: u64,
    metadata_length: u64,
    index_offset: u64,
    index_length: u64,
    payload_offset: u64,
    payload_length: u64,
    root_hash: [u8; HASH_SIZE],
    signature_offset: u64,
    signature_length: u64,
}

fn parse_file_header(bytes: &[u8; HEADER_SIZE], file_length: u64) -> Result<FileHeader> {
    if &bytes[..4] != MAGIC || u32_at(bytes, 4) != VERSION {
        return Err(invalid("invalid QIF v2 file header"));
    }
    if u32_at(bytes, 8) & !KNOWN_FLAGS != 0 || u32_at(bytes, 12) != HEADER_SIZE as u32 {
        return Err(invalid("unsupported QIF v2 header flags or size"));
    }
    if u32_at(bytes, 16) != HASH_SHA256 || u32_at(bytes, 20) != INDEX_VERSION {
        return Err(invalid("unsupported QIF v2 header algorithm or index"));
    }
    let signature_offset = u64_at(bytes, 120);
    let signature_length = u64_at(bytes, 128);
    if u64_at(bytes, 136) != 0 {
        return Err(invalid("reserved header fields must be zero"));
    }
    let sections = [
        ("manifest", u64_at(bytes, 24), u64_at(bytes, 32)),
        ("metadata", u64_at(bytes, 40), u64_at(bytes, 48)),
        ("index", u64_at(bytes, 56), u64_at(bytes, 64)),
        ("payload", u64_at(bytes, 72), u64_at(bytes, 80)),
    ];
    for (name, offset, length) in sections {
        if offset < HEADER_SIZE as u64 || !offset.is_multiple_of(ALIGNMENT as u64) {
            return Err(invalid(format!("{name} offset is invalid")));
        }
        if offset.checked_add(length).is_none() || offset + length > file_length {
            return Err(invalid(format!("{name} range exceeds file")));
        }
        if name != "payload" && length > MAX_HEADER_SECTION as u64 {
            return Err(invalid(format!("{name} exceeds the size limit")));
        }
    }
    if sections[2].2 > MAX_INDEX_BYTES as u64 {
        return Err(invalid("index exceeds the size limit"));
    }
    let mut ranges = sections
        .iter()
        .map(|(name, offset, length)| (*name, *offset, offset + length))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|(_, start, _)| *start);
    if ranges.windows(2).any(|pair| pair[1].1 < pair[0].2) {
        return Err(invalid("sections overlap"));
    }
    if signature_length == 0 {
        if signature_offset != 0 {
            return Err(invalid("signature offset is nonzero with zero length"));
        }
    } else {
        if signature_length != SIGNATURE_LENGTH as u64
            || signature_offset < HEADER_SIZE as u64
            || !signature_offset.is_multiple_of(ALIGNMENT as u64)
        {
            return Err(invalid("invalid QIF2 signature range"));
        }
        let signature_end = signature_offset
            .checked_add(signature_length)
            .ok_or_else(|| invalid("signature range overflows"))?;
        if signature_end > file_length
            || sections.iter().any(|(_, start, length)| {
                signature_offset < start.saturating_add(*length) && *start < signature_end
            })
        {
            return Err(invalid("signature overlaps another section"));
        }
    }
    if sections[3].1 + sections[3].2 != file_length {
        return Err(invalid("unexpected bytes outside payload section"));
    }
    Ok(FileHeader {
        manifest_offset: sections[0].1,
        manifest_length: sections[0].2,
        metadata_offset: sections[1].1,
        metadata_length: sections[1].2,
        index_offset: sections[2].1,
        index_length: sections[2].2,
        payload_offset: sections[3].1,
        payload_length: sections[3].2,
        root_hash: bytes[88..120].try_into().unwrap(),
        signature_offset,
        signature_length,
    })
}

fn read_section(file: &mut File, offset: u64, length: u64) -> Result<Vec<u8>> {
    let length = usize::try_from(length).map_err(|_| invalid("section length overflows usize"))?;
    let mut bytes = vec![0_u8; length];
    read_at(file, offset, &mut bytes)?;
    Ok(bytes)
}

fn read_at(file: &mut File, offset: u64, bytes: &mut [u8]) -> Result<()> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| crate::FormatError::Io(format!("cannot seek QIF v2: {error}")))?;
    file.read_exact(bytes)
        .map_err(|error| crate::FormatError::Io(format!("cannot read QIF v2: {error}")))
}

pub(crate) fn has_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 8
        && &bytes[..4] == MAGIC
        && u32::from_le_bytes(bytes[4..8].try_into().unwrap()) == VERSION
}

pub(crate) fn open_packed(bytes: Vec<u8>) -> Result<PackedTensorReader> {
    let reader = QifV2Reader::open(&bytes)?;
    let metadata = metadata_to_gguf(reader.metadata())?;
    let compressed = reader.tensors.iter().any(|tensor| {
        let physical = resolve_tensor(&reader.tensors, tensor.clone()).ok();
        tensor.codec != CODEC_NONE || physical.is_some_and(|value| value.chunk_count != 1)
    });
    let mut materialized = Vec::new();
    let mut materialized_offsets = BTreeMap::new();
    if compressed {
        for tensor in &reader.tensors {
            let offset = materialized.len();
            materialized.extend_from_slice(&reader.tensor_data(&tensor.name)?);
            materialized_offsets.insert(tensor.tensor_id, (offset, tensor.raw_length));
        }
    }
    let data_start = if compressed { 0 } else { reader.payload_offset };
    let embedded_manifest = reader.manifest.clone();
    let index_tensors = reader.tensors.clone();
    let index_chunks = reader.chunks.clone();
    drop(reader);
    let ggml = match metadata.get("quantinfer.qif.ggml_dimension_order") {
        Some(GgufValue::Bool(value)) => *value,
        Some(_) => return Err(invalid("GGML dimension order metadata must be boolean")),
        None => false,
    };
    let tensors = index_tensors
        .iter()
        .map(|tensor| {
            let storage = tensor_storage_info(&index_tensors, &index_chunks, tensor)?;
            let mut shape = tensor
                .shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .map_err(|_| invalid("tensor dimension overflows usize"))
                })
                .collect::<Result<Vec<_>>>()?;
            if ggml {
                shape.reverse();
            }
            Ok(GgufTensorInfo {
                name: tensor.name.clone(),
                shape,
                dtype: dtype_to_ggml(tensor.dtype)?,
                offset: if compressed {
                    materialized_offsets
                        .get(&tensor.tensor_id)
                        .map(|(offset, _)| *offset as u64)
                        .ok_or_else(|| invalid("materialized tensor is missing"))?
                } else {
                    storage.0
                },
                size: if compressed {
                    usize::try_from(
                        materialized_offsets
                            .get(&tensor.tensor_id)
                            .map(|(_, length)| *length)
                            .ok_or_else(|| invalid("materialized tensor is missing"))?,
                    )
                    .map_err(|_| invalid("materialized tensor length overflows usize"))?
                } else {
                    storage.1
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PackedTensorReader {
        bytes: if compressed { materialized } else { bytes },
        metadata,
        tensors,
        data_start,
        embedded_manifest: Some(embedded_manifest),
    })
}

fn tensor_storage_info(
    tensors: &[QifV2TensorInfo],
    chunks: &[ChunkInfo],
    tensor: &QifV2TensorInfo,
) -> Result<(u64, usize)> {
    if let Some(alias) = tensor.alias_of {
        let target = tensors
            .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
            .ok_or_else(|| invalid("alias target is out of range"))?;
        return tensor_storage_info(tensors, chunks, target);
    }
    let offset = chunks
        .get(tensor.chunk_start)
        .ok_or_else(|| invalid("tensor chunk range is invalid"))?
        .stored_offset;
    let length = usize::try_from(tensor.stored_length)
        .map_err(|_| invalid("tensor length overflows usize"))?;
    Ok((offset, length))
}

fn parse_control(
    manifest_bytes: &[u8],
    metadata_bytes: &[u8],
    index_bytes: &[u8],
    payload_length: u64,
    root_hash: [u8; HASH_SIZE],
    signature: Option<&[u8]>,
) -> Result<ParsedControl> {
    let mut root = Sha256::new();
    root.update(manifest_bytes);
    root.update(metadata_bytes);
    root.update(index_bytes);
    if root.finalize()[..] != root_hash {
        return Err(invalid("root hash mismatch"));
    }
    if let Some(signature) = signature {
        super::signature::verify(&root_hash, signature)?;
    }
    let manifest_text =
        std::str::from_utf8(manifest_bytes).map_err(|_| invalid("manifest is not UTF-8"))?;
    let manifest_value: serde_json::Value = serde_json::from_slice(manifest_bytes)
        .map_err(|error| invalid(format!("invalid manifest JSON: {error}")))?;
    if json_depth(&manifest_value) > MAX_MANIFEST_DEPTH {
        return Err(invalid("manifest nesting depth exceeds the limit"));
    }
    let manifest = ModelManifest::from_json(manifest_text)
        .map_err(|error| invalid(format!("invalid manifest: {error}")))?;
    validate_manifest_limits(&manifest)?;
    let canonical = serde_json::to_vec(&manifest)
        .map_err(|error| invalid(format!("cannot canonicalize manifest: {error}")))?;
    if canonical != manifest_bytes {
        return Err(invalid("manifest is not in the QIF v2 canonical JSON form"));
    }
    let metadata = super::metadata::decode(metadata_bytes)?;
    let (tensors, chunks) = parse_index(index_bytes, payload_length, &manifest)?;
    for tensor in &tensors {
        let manifest_tensor = manifest
            .tensors
            .iter()
            .find(|candidate| candidate.name == tensor.name)
            .ok_or_else(|| invalid(format!("index contains unknown tensor {}", tensor.name)))?;
        let checksum = hex_encode(&tensor.checksum_sha256);
        if manifest_tensor.checksum_sha256.as_deref() != Some(checksum.as_str()) {
            return Err(invalid(format!(
                "tensor {} checksum disagrees with manifest",
                tensor.name
            )));
        }
        validate_tensor_descriptor(tensor, manifest_tensor)?;
    }
    if tensors.len() != manifest.tensors.len() {
        return Err(invalid("manifest and index tensor counts differ"));
    }
    for tensor in &tensors {
        if let Some(alias) = tensor.alias_of {
            let target = tensors
                .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("alias target is out of range"))?;
            if target.alias_of.is_some()
                || target.chunk_count == 0
                || tensor.shape != target.shape
                || tensor.dtype != target.dtype
                || tensor.physical_encoding != target.physical_encoding
                || tensor.quantization_scheme != target.quantization_scheme
                || tensor.layout != target.layout
                || tensor.codec != target.codec
                || tensor.raw_length != target.raw_length
                || tensor.stored_length != target.stored_length
                || tensor.checksum_sha256 != target.checksum_sha256
            {
                return Err(invalid(format!(
                    "tensor {} alias metadata disagrees with target",
                    tensor.name
                )));
            }
        }
    }
    Ok((manifest, metadata, tensors, chunks))
}

fn json_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(values) => values
            .iter()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        serde_json::Value::Object(values) => values
            .values()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => 0,
    }
}

fn validate_manifest_limits(manifest: &ModelManifest) -> Result<()> {
    let mut strings = vec![
        &manifest.model_id,
        &manifest.architecture,
        &manifest.source_format,
    ];
    if let Some(value) = &manifest.source_revision {
        strings.push(value);
    }
    if let Some(value) = &manifest.tokenizer_revision {
        strings.push(value);
    }
    for tensor in &manifest.tensors {
        strings.push(&tensor.name);
        strings.push(&tensor.descriptor.physical_layout);
        strings.push(&tensor.descriptor.packing);
        strings.extend(tensor.descriptor.backend_requirements.iter());
    }
    for variant in &manifest.execution_variants {
        strings.push(&variant.name);
        strings.push(&variant.dtype);
        strings.push(&variant.runtime_provider);
        strings.push(&variant.kernel_provider);
    }
    strings.extend(manifest.kernel_requirements.operations.iter());
    if let Some(value) = &manifest.compatibility.minimum_quantinfer_version {
        strings.push(value);
    }
    strings.extend(manifest.compatibility.required_features.iter());
    if strings
        .iter()
        .any(|value| value.len() > MAX_MANIFEST_STRING_BYTES)
    {
        return Err(invalid("manifest string exceeds the size limit"));
    }
    if manifest.tensors.len() > MAX_TENSORS || manifest.execution_variants.len() > MAX_TENSORS {
        return Err(invalid("manifest item count exceeds the size limit"));
    }
    Ok(())
}

fn parse_index(
    bytes: &[u8],
    payload_length: u64,
    manifest: &ModelManifest,
) -> Result<(Vec<QifV2TensorInfo>, Vec<ChunkInfo>)> {
    if bytes.len() < INDEX_HEADER_SIZE || &bytes[..4] != INDEX_MAGIC {
        return Err(invalid("invalid index header"));
    }
    if u32_at(bytes, 4) != INDEX_VERSION {
        return Err(invalid("unsupported index version"));
    }
    let tensor_count =
        usize::try_from(u32_at(bytes, 8)).map_err(|_| invalid("tensor count overflows usize"))?;
    let chunk_count =
        usize::try_from(u32_at(bytes, 12)).map_err(|_| invalid("chunk count overflows usize"))?;
    if tensor_count > MAX_TENSORS
        || chunk_count > MAX_CHUNKS
        || tensor_count != manifest.tensors.len()
    {
        return Err(invalid("index item count is invalid"));
    }
    let strings_len = usize::try_from(u64_at(bytes, 16))
        .map_err(|_| invalid("string table length overflows usize"))?;
    let shapes_len = usize::try_from(u64_at(bytes, 24))
        .map_err(|_| invalid("shape table length overflows usize"))?;
    let records_len = usize::try_from(u64_at(bytes, 32))
        .map_err(|_| invalid("tensor record length overflows usize"))?;
    let chunks_len = usize::try_from(u64_at(bytes, 40))
        .map_err(|_| invalid("chunk record length overflows usize"))?;
    if records_len != tensor_count * TENSOR_RECORD_SIZE
        || chunks_len != chunk_count * CHUNK_RECORD_SIZE
    {
        return Err(invalid("index record lengths do not match item counts"));
    }
    let strings_start = INDEX_HEADER_SIZE;
    let shapes_start = strings_start
        .checked_add(strings_len)
        .ok_or_else(|| invalid("string table overflows"))?;
    let records_start = shapes_start
        .checked_add(shapes_len)
        .ok_or_else(|| invalid("shape table overflows"))?;
    let chunks_start = records_start
        .checked_add(records_len)
        .ok_or_else(|| invalid("tensor records overflow"))?;
    let end = chunks_start
        .checked_add(chunks_len)
        .ok_or_else(|| invalid("chunk records overflow"))?;
    if end != bytes.len() {
        return Err(invalid("index has trailing or truncated bytes"));
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    for index in 0..chunk_count {
        let offset = chunks_start + index * CHUNK_RECORD_SIZE;
        let hash: [u8; HASH_SIZE] = bytes[offset + 40..offset + 72].try_into().unwrap();
        let chunk = ChunkInfo {
            shard_id: u32_at(bytes, offset),
            codec: u32_at(bytes, offset + 4),
            raw_offset: u64_at(bytes, offset + 8),
            raw_length: u64_at(bytes, offset + 16),
            stored_offset: u64_at(bytes, offset + 24),
            stored_length: u64_at(bytes, offset + 32),
            checksum_sha256: hash,
        };
        if chunk.shard_id != 0 || !matches!(chunk.codec, CODEC_NONE | CODEC_ZSTD | CODEC_LZ4) {
            return Err(invalid("unsupported QIF v2 single-file chunk codec"));
        }
        if chunk.codec == CODEC_NONE && chunk.stored_length != chunk.raw_length {
            return Err(invalid(
                "uncompressed chunk length disagrees with raw length",
            ));
        }
        if chunk
            .stored_offset
            .checked_add(chunk.stored_length)
            .is_none()
            || chunk.stored_offset + chunk.stored_length > payload_length
            || chunk.raw_offset.checked_add(chunk.raw_length).is_none()
            || !chunk.stored_offset.is_multiple_of(ALIGNMENT as u64)
        {
            return Err(invalid("chunk range exceeds the payload section"));
        }
        chunks.push(chunk);
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut names = BTreeSet::new();
    let mut occupied = Vec::<(u64, u64)>::new();
    for index in 0..tensor_count {
        let offset = records_start + index * TENSOR_RECORD_SIZE;
        let tensor_id = u64_at(bytes, offset);
        if tensor_id != index as u64 {
            return Err(invalid("tensor ids must be contiguous"));
        }
        let name = read_string_ref(bytes, strings_start, strings_len, u32_at(bytes, offset + 8))?;
        if !names.insert(name.clone()) {
            return Err(invalid("tensor names must be unique"));
        }
        let shape = read_shape_ref(bytes, shapes_start, shapes_len, u32_at(bytes, offset + 12))?;
        if shape.len() > MAX_RANK {
            return Err(invalid("tensor rank exceeds the limit"));
        }
        let chunk_start = usize::try_from(u32_at(bytes, offset + 40)).unwrap();
        let chunk_count_for_tensor = usize::try_from(u32_at(bytes, offset + 44)).unwrap();
        let end_chunk = chunk_start
            .checked_add(chunk_count_for_tensor)
            .ok_or_else(|| invalid("tensor chunk range overflows"))?;
        if end_chunk > chunks.len() {
            return Err(invalid("tensor chunk range is invalid"));
        }
        let raw_length = u64_at(bytes, offset + 48);
        let stored_length = u64_at(bytes, offset + 56);
        let data_offset = u64_at(bytes, offset + 64);
        let checksum: [u8; HASH_SIZE] = bytes[offset + 72..offset + 104].try_into().unwrap();
        let alias_raw = u64_at(bytes, offset + 104);
        let alias_of = if alias_raw == ALIAS_NONE {
            None
        } else {
            Some(alias_raw)
        };
        if let Some(alias) = alias_of {
            if alias >= tensor_count as u64 || alias >= tensor_id || chunk_count_for_tensor != 0 {
                return Err(invalid("tensor alias target is invalid"));
            }
        } else {
            if chunk_count_for_tensor == 0 {
                return Err(invalid(
                    "non-alias tensor must reference at least one chunk",
                ));
            }
            let mut raw_sum = 0u64;
            let mut stored_sum = 0u64;
            for (position, chunk) in chunks[chunk_start..end_chunk].iter().enumerate() {
                if position == 0 && chunk.stored_offset != data_offset {
                    return Err(invalid("tensor data offset disagrees with first chunk"));
                }
                if position > 0 {
                    let previous = &chunks[chunk_start + position - 1];
                    let previous_end = previous
                        .stored_offset
                        .checked_add(previous.stored_length)
                        .ok_or_else(|| invalid("tensor stored offset overflows"))?;
                    if chunk.stored_offset != align_u64(previous_end)? {
                        return Err(invalid("tensor chunk offsets are not aligned"));
                    }
                }
                if chunk.raw_offset != raw_sum {
                    return Err(invalid("tensor chunk offsets are not contiguous"));
                }
                raw_sum = raw_sum
                    .checked_add(chunk.raw_length)
                    .ok_or_else(|| invalid("tensor raw length overflows"))?;
                stored_sum = stored_sum
                    .checked_add(chunk.stored_length)
                    .ok_or_else(|| invalid("tensor stored length overflows"))?;
            }
            if raw_sum != raw_length || stored_sum != stored_length {
                return Err(invalid("tensor lengths disagree with chunks"));
            }
            let tensor_codec = u32_at(bytes, offset + 32);
            if tensor_codec != chunks[chunk_start].codec
                || chunks[chunk_start..end_chunk]
                    .iter()
                    .any(|chunk| chunk.codec != tensor_codec)
            {
                return Err(invalid("tensor codec disagrees with its chunks"));
            }
            if raw_sum != 0 && chunks[chunk_start].raw_offset != 0 {
                return Err(invalid(
                    "tensor raw chunks must start at logical offset zero",
                ));
            }
            for chunk in &chunks[chunk_start..end_chunk] {
                let end = chunk
                    .stored_offset
                    .checked_add(chunk.stored_length)
                    .ok_or_else(|| invalid("tensor range overflows"))?;
                occupied.push((chunk.stored_offset, end));
            }
        }
        tensors.push(QifV2TensorInfo {
            tensor_id,
            name,
            shape,
            dtype: u32_at(bytes, offset + 16),
            physical_encoding: u32_at(bytes, offset + 20),
            quantization_scheme: u32_at(bytes, offset + 24),
            layout: u32_at(bytes, offset + 28),
            codec: u32_at(bytes, offset + 32),
            raw_length,
            stored_length,
            checksum_sha256: checksum,
            alias_of,
            chunk_start,
            chunk_count: chunk_count_for_tensor,
        });
    }
    occupied.sort_unstable_by_key(|(start, _)| *start);
    if occupied.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(invalid("tensor chunks overlap"));
    }
    Ok((tensors, chunks))
}

pub(crate) fn validate_tensor_descriptor(
    tensor: &QifV2TensorInfo,
    manifest: &crate::TensorManifest,
) -> Result<()> {
    let descriptor = &manifest.descriptor;
    let expected_dtype = match descriptor.dtype {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::Q4_0 => 4,
        DType::Q4_K => 5,
        DType::Q5_K => 6,
        DType::Q6_K => 7,
        DType::Q8_0 => 8,
    };
    if tensor.dtype != expected_dtype || tensor.physical_encoding != expected_dtype {
        return Err(invalid(format!(
            "tensor {} dtype and encoding disagree with manifest",
            tensor.name
        )));
    }
    let expected_scheme = match descriptor.quantization {
        quantinfer_core::QuantizationScheme::None => 0,
        quantinfer_core::QuantizationScheme::Q8_0 => 1,
        quantinfer_core::QuantizationScheme::Q4_0 => 2,
        quantinfer_core::QuantizationScheme::Q4_K => 3,
        quantinfer_core::QuantizationScheme::Q5_K => 4,
        quantinfer_core::QuantizationScheme::Q6_K => 5,
        quantinfer_core::QuantizationScheme::Mixed => 6,
    };
    if tensor.quantization_scheme != expected_scheme {
        return Err(invalid(format!(
            "tensor {} quantization disagrees with manifest",
            tensor.name
        )));
    }
    let expected_layout = match descriptor.physical_layout.as_str() {
        "row_major" => 1,
        "output_rows" => 2,
        value => return Err(invalid(format!("unsupported tensor layout {value}"))),
    };
    if tensor.layout != expected_layout {
        return Err(invalid(format!(
            "tensor {} layout disagrees with manifest",
            tensor.name
        )));
    }
    let expected_shape = descriptor
        .logical_shape
        .dims()
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| invalid("tensor dimension overflows u64"))
        })
        .collect::<Result<Vec<_>>>()?;
    if tensor.shape != expected_shape {
        return Err(invalid(format!(
            "tensor {} shape disagrees with manifest",
            tensor.name
        )));
    }
    let expected_dtype = dtype_to_ggml(tensor.dtype)?;
    let expected_size = crate::tensor_byte_size(expected_dtype, descriptor.logical_shape.dims())?;
    if tensor.raw_length != expected_size as u64 {
        return Err(invalid(format!(
            "tensor {} length disagrees with descriptor",
            tensor.name
        )));
    }
    Ok(())
}

fn read_string_ref(bytes: &[u8], start: usize, length: usize, offset: u32) -> Result<String> {
    let offset = usize::try_from(offset).unwrap();
    let prefix = start
        .checked_add(offset)
        .ok_or_else(|| invalid("string offset overflows"))?;
    let end = prefix
        .checked_add(4)
        .ok_or_else(|| invalid("string length overflows"))?;
    if end > start + length {
        return Err(invalid("string reference is outside the table"));
    }
    let string_length =
        usize::try_from(u32::from_le_bytes(bytes[prefix..end].try_into().unwrap())).unwrap();
    let value_start = end;
    let value_end = value_start
        .checked_add(string_length)
        .ok_or_else(|| invalid("string range overflows"))?;
    if value_end > start + length {
        return Err(invalid("string reference is truncated"));
    }
    String::from_utf8(bytes[value_start..value_end].to_vec())
        .map_err(|_| invalid("index string is not UTF-8"))
}

fn read_shape_ref(bytes: &[u8], start: usize, length: usize, offset: u32) -> Result<Vec<u64>> {
    let offset = usize::try_from(offset).unwrap();
    let prefix = start
        .checked_add(offset)
        .ok_or_else(|| invalid("shape offset overflows"))?;
    let end = prefix
        .checked_add(4)
        .ok_or_else(|| invalid("shape rank overflows"))?;
    if end > start + length {
        return Err(invalid("shape reference is outside the table"));
    }
    let rank = usize::try_from(u32::from_le_bytes(bytes[prefix..end].try_into().unwrap())).unwrap();
    if rank > MAX_RANK {
        return Err(invalid("shape rank exceeds the limit"));
    }
    let dimensions_end = end
        .checked_add(rank * 8)
        .ok_or_else(|| invalid("shape dimensions overflow"))?;
    if dimensions_end > start + length {
        return Err(invalid("shape reference is truncated"));
    }
    let mut shape = Vec::with_capacity(rank);
    for index in 0..rank {
        let offset = end + index * 8;
        let dimension = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        if dimension == 0 {
            return Err(invalid("tensor dimensions must be greater than zero"));
        }
        shape.push(dimension);
    }
    Ok(shape)
}

fn dtype_to_ggml(value: u32) -> Result<GgmlType> {
    match value {
        1 => Ok(GgmlType::F32),
        2 => Ok(GgmlType::F16),
        3 => Ok(GgmlType::BF16),
        4 => Ok(GgmlType::Q4_0),
        5 => Ok(GgmlType::Q4_K),
        6 => Ok(GgmlType::Q5_K),
        7 => Ok(GgmlType::Q6_K),
        8 => Ok(GgmlType::Q8_0),
        value => Err(invalid(format!("unsupported QIF v2 dtype {value}"))),
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn align_u64(offset: u64) -> Result<u64> {
    let alignment = ALIGNMENT as u64;
    offset
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| invalid("offset alignment overflows"))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
