use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DuplicateFreeJson, FormatError, ModelManifest, Result};

use super::chunk::Chunk as PackageChunk;
use super::codec::{compress, decompress};
use super::metadata::MetadataValue;
use super::payload::{validate_payload, PayloadVerifier};
use super::reader::{read_shape_ref, read_string_ref, validate_tensor_descriptor, TensorInfo};
use super::writer::Codec;
use super::{
    align, canonicalize_manifest_layouts, invalid, ALIAS_NONE, ALIGNMENT, CODEC_LZ4, CODEC_NONE,
    CODEC_ZSTD, HASH_SIZE, INDEX_MAGIC, INDEX_VERSION, MAX_CHUNKS, MAX_INDEX_BYTES, MAX_RANK,
    MAX_TENSORS,
};

const PACKAGE_SCHEMA_VERSION: u32 = 1;
const PACKAGE_PROFILE: &str = "QIF-PACKAGE";
const LEGACY_PACKAGE_PROFILE: &str = "QIF2-PACKAGE";
const MAX_PACKAGE_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
const INDEX_HEADER_SIZE: usize = 48;
const TENSOR_RECORD_SIZE: usize = 120;
const CHUNK_RECORD_SIZE: usize = 72;
const DEFAULT_SHARD_LIMIT: usize = 1024 * 1024 * 1024;
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackageManifest {
    schema_version: u32,
    profile: String,
    model: ModelManifest,
    metadata_path: String,
    metadata_sha256: String,
    index_path: String,
    index_sha256: String,
    shards: Vec<ShardManifest>,
    root_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShardManifest {
    path: String,
    size: u64,
    sha256: String,
    mesh_axis: u32,
    rank: u32,
    coordinate: Vec<u32>,
    tensors: Vec<String>,
}

#[derive(Clone, Copy)]
struct PhysicalTensor {
    chunk_start: u32,
    chunk_count: u32,
    raw_length: usize,
    stored_length: usize,
    data_offset: usize,
    hash: [u8; HASH_SIZE],
    codec: u32,
    dtype: u32,
    encoding: u32,
    scheme: u32,
    layout: u32,
}

/// Write a QIF-PACKAGE directory with independently readable shard files.
pub fn write_qif_v2_package<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output_dir: impl AsRef<Path>,
    max_shard_bytes: Option<usize>,
) -> Result<()> {
    write_qif_v2_package_with_codec(
        manifest,
        metadata,
        tensor_bytes,
        output_dir,
        max_shard_bytes,
        Codec::None,
    )
}

pub fn write_qif_v2_package_with_codec<'a>(
    mut manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output_dir: impl AsRef<Path>,
    max_shard_bytes: Option<usize>,
    codec: Codec,
) -> Result<()> {
    canonicalize_manifest_layouts(&mut manifest);
    manifest.validate()?;
    super::reader::validate_manifest_limits(&manifest)?;
    let output_dir = output_dir.as_ref();
    if output_dir.exists() {
        return Err(invalid(format!(
            "package output already exists: {}",
            output_dir.display()
        )));
    }
    let max_shard_bytes = max_shard_bytes.unwrap_or(DEFAULT_SHARD_LIMIT);
    if max_shard_bytes == 0 {
        return Err(invalid("package shard size must be greater than zero"));
    }
    let parent = output_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(io_error)?;
    let stage_id = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".qif-stage-{}-{stage_id}", std::process::id()));
    fs::create_dir(&temporary).map_err(io_error)?;
    let result = write_package_contents(
        &mut manifest,
        metadata,
        &mut tensor_bytes,
        &temporary,
        max_shard_bytes,
        codec.id(),
    )
    .and_then(|()| PackageReader::open(&temporary)?.verify())
    .and_then(|()| fs::rename(&temporary, output_dir).map_err(io_error));
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn write_package_contents<'a>(
    manifest: &mut ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: &mut impl FnMut(&str) -> Result<&'a [u8]>,
    output_dir: &Path,
    max_shard_bytes: usize,
    codec: u32,
) -> Result<()> {
    let metadata_bytes = super::metadata::encode(metadata)?;
    write_atomic_file(&output_dir.join("metadata.qmb"), &metadata_bytes)?;

    let mut payloads = Vec::with_capacity(manifest.tensors.len());
    for tensor in &mut manifest.tensors {
        let bytes = tensor_bytes(&tensor.name)?;
        let expected = package_payload_size(&tensor.descriptor)?;
        if bytes.len() != expected {
            return Err(invalid(format!(
                "tensor {} has {} bytes, expected {expected}",
                tensor.name,
                bytes.len()
            )));
        }
        validate_payload(tensor.descriptor.dtype, bytes, &tensor.name)?;
        tensor.checksum_sha256 = Some(crate::sha256_hex(bytes));
        payloads.push(bytes);
    }

    let mut strings = Vec::new();
    let mut shapes = Vec::new();
    let mut records = Vec::new();
    let mut chunks = Vec::new();
    let mut dedup = BTreeMap::<[u8; HASH_SIZE], Vec<usize>>::new();
    let mut physical = Vec::<Option<PhysicalTensor>>::with_capacity(manifest.tensors.len());
    let mut shard_bytes = vec![Vec::<u8>::new()];
    let mut shard_tensors = vec![BTreeSet::<String>::new()];
    for (tensor_id, (tensor, payload)) in manifest.tensors.iter().zip(payloads.iter()).enumerate() {
        let name_offset = strings.len();
        strings.extend_from_slice(
            &(u32::try_from(tensor.name.len()).map_err(|_| invalid("tensor name is too large"))?)
                .to_le_bytes(),
        );
        strings.extend_from_slice(tensor.name.as_bytes());
        let shape_offset = shapes.len();
        let dims = tensor.descriptor.logical_shape.dims();
        if dims.len() > MAX_RANK {
            return Err(invalid("tensor rank exceeds the limit"));
        }
        shapes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dimension in dims {
            shapes.extend_from_slice(
                &(u64::try_from(*dimension)
                    .map_err(|_| invalid("tensor dimension overflows u64"))?)
                .to_le_bytes(),
            );
        }
        let (dtype, scheme, layout) = descriptor_ids(&tensor.descriptor)?;
        let raw_hash: [u8; HASH_SIZE] = Sha256::digest(payload).into();
        let alias_target = dedup.get(&raw_hash).and_then(|candidates| {
            candidates.iter().copied().find(|candidate| {
                let prior = physical[*candidate].expect("dedup candidate is physical");
                manifest.tensors[*candidate].descriptor == tensor.descriptor
                    && prior.dtype == dtype
                    && prior.encoding == dtype
                    && prior.scheme == scheme
                    && prior.layout == layout
                    && payloads[*candidate] == *payload
            })
        });
        if let Some(target) = alias_target {
            let prior = physical[target].expect("alias target is physical");
            let chunk_bytes = &chunks[prior.chunk_start as usize * CHUNK_RECORD_SIZE
                ..(prior.chunk_start as usize + prior.chunk_count as usize) * CHUNK_RECORD_SIZE];
            for chunk in chunk_bytes.as_chunks::<CHUNK_RECORD_SIZE>().0 {
                let shard_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) as usize;
                shard_tensors[shard_id].insert(tensor.name.clone());
            }
            records.extend_from_slice(&encode_tensor_record(
                tensor_id as u64,
                u32::try_from(name_offset)
                    .map_err(|_| invalid("name table offset overflows u32"))?,
                u32::try_from(shape_offset)
                    .map_err(|_| invalid("shape table offset overflows u32"))?,
                dtype,
                dtype,
                scheme,
                layout,
                prior.codec,
                0,
                0,
                prior.raw_length,
                prior.stored_length,
                prior.data_offset,
                prior.hash,
                target as u64,
            ));
            physical.push(None);
        } else {
            let chunk_start = chunks.len() / CHUNK_RECORD_SIZE;
            let mut raw_offset = 0usize;
            let mut stored_total = 0usize;
            let mut first_stored_offset = None;
            while raw_offset < payload.len() {
                if chunks.len() / CHUNK_RECORD_SIZE >= MAX_CHUNKS {
                    return Err(invalid("package chunk count exceeds the limit"));
                }
                let remaining = payload.len() - raw_offset;
                let raw_limit = if codec == CODEC_NONE {
                    usize::MAX
                } else {
                    super::codec::DEFAULT_CHUNK_BYTES
                };
                let mut take = remaining.min(max_shard_bytes).min(raw_limit);
                let mut stored = compress(codec, &payload[raw_offset..raw_offset + take])?;
                while stored.len() > max_shard_bytes {
                    if take == 1 {
                        return Err(invalid("shard size limit is too small for codec framing"));
                    }
                    take /= 2;
                    stored = compress(codec, &payload[raw_offset..raw_offset + take])?;
                }
                if shard_bytes.last().is_some_and(|shard| {
                    !shard.is_empty()
                        && align(shard.len())
                            .ok()
                            .and_then(|start| start.checked_add(stored.len()))
                            .is_none_or(|end| end > max_shard_bytes)
                }) {
                    shard_bytes.push(Vec::new());
                    shard_tensors.push(BTreeSet::new());
                }
                let shard_id = shard_bytes.len() - 1;
                let shard = &mut shard_bytes[shard_id];
                let stored_offset = align(shard.len())?;
                shard.resize(stored_offset, 0);
                shard.extend_from_slice(&stored);
                first_stored_offset.get_or_insert(stored_offset);
                let stored_hash: [u8; HASH_SIZE] = Sha256::digest(&stored).into();
                chunks.extend_from_slice(&encode_chunk(
                    shard_id as u32,
                    codec,
                    raw_offset,
                    take,
                    stored_offset,
                    stored.len(),
                    stored_hash,
                ));
                shard_tensors[shard_id].insert(tensor.name.clone());
                raw_offset += take;
                stored_total += stored.len();
            }
            let info = PhysicalTensor {
                chunk_start: u32::try_from(chunk_start)
                    .map_err(|_| invalid("chunk index overflows u32"))?,
                chunk_count: u32::try_from(chunks.len() / CHUNK_RECORD_SIZE - chunk_start)
                    .map_err(|_| invalid("chunk count overflows u32"))?,
                raw_length: payload.len(),
                stored_length: stored_total,
                data_offset: first_stored_offset
                    .ok_or_else(|| invalid("package chunk is missing"))?,
                hash: raw_hash,
                codec,
                dtype,
                encoding: dtype,
                scheme,
                layout,
            };
            records.extend_from_slice(&encode_tensor_record(
                tensor_id as u64,
                u32::try_from(name_offset)
                    .map_err(|_| invalid("name table offset overflows u32"))?,
                u32::try_from(shape_offset)
                    .map_err(|_| invalid("shape table offset overflows u32"))?,
                dtype,
                dtype,
                scheme,
                layout,
                codec,
                info.chunk_start,
                info.chunk_count,
                info.raw_length,
                info.stored_length,
                info.data_offset,
                info.hash,
                ALIAS_NONE,
            ));
            physical.push(Some(info));
            dedup.entry(raw_hash).or_default().push(tensor_id);
        }
    }
    if chunks.len() / CHUNK_RECORD_SIZE > MAX_CHUNKS {
        return Err(invalid("package chunk count exceeds the limit"));
    }
    let index = encode_index(
        manifest.tensors.len(),
        chunks.len() / CHUNK_RECORD_SIZE,
        &strings,
        &shapes,
        &records,
        &chunks,
    )?;
    write_atomic_file(&output_dir.join("index.qifi"), &index)?;
    let mut shards = Vec::with_capacity(shard_bytes.len());
    for (shard_id, bytes) in shard_bytes.into_iter().enumerate() {
        let path = format!("shard-{shard_id:05}.qifd");
        write_atomic_file(&output_dir.join(&path), &bytes)?;
        shards.push(ShardManifest {
            path,
            size: bytes.len() as u64,
            sha256: crate::sha256_hex(&bytes),
            mesh_axis: 0,
            rank: 1,
            coordinate: vec![shard_id as u32],
            tensors: shard_tensors[shard_id].iter().cloned().collect(),
        });
    }
    let mut package = PackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        profile: PACKAGE_PROFILE.to_owned(),
        model: manifest.clone(),
        metadata_path: "metadata.qmb".to_owned(),
        metadata_sha256: crate::sha256_hex(&metadata_bytes),
        index_path: "index.qifi".to_owned(),
        index_sha256: crate::sha256_hex(&index),
        shards,
        root_hash: String::new(),
    };
    let unsigned = serde_json::to_vec(&package).map_err(json_error)?;
    package.root_hash = package_root_hash(&unsigned, &metadata_bytes, &index, &package.shards);
    let package_bytes = serde_json::to_vec(&package).map_err(json_error)?;
    if package_bytes.len() > MAX_PACKAGE_MANIFEST_BYTES {
        return Err(invalid("package manifest exceeds the size limit"));
    }
    write_atomic_file(&output_dir.join("qif.json"), &package_bytes)
}

struct OpenShard {
    name: String,
    file: File,
}

pub struct PackageReader {
    root: PathBuf,
    manifest: ModelManifest,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    pub(super) chunks: Vec<PackageChunk>,
    open_shard: Option<OpenShard>,
    entries: BTreeMap<String, ShardManifest>,
    pub(super) artifact_length: u64,
    pub(super) control_length: usize,
}

impl PackageReader {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = fs::canonicalize(root.as_ref()).map_err(io_error)?;
        let package_path = resolve_path(&root, "qif.json")?;
        let package_bytes = read_bounded_file(
            &package_path,
            MAX_PACKAGE_MANIFEST_BYTES,
            "QIF package manifest",
        )?;
        super::validate_json_depth(&package_bytes)?;
        let package_value: DuplicateFreeJson =
            serde_json::from_slice(&package_bytes).map_err(json_error)?;
        let package: PackageManifest =
            serde_json::from_value(package_value.0).map_err(json_error)?;
        if package.schema_version != PACKAGE_SCHEMA_VERSION
            || !matches!(
                package.profile.as_str(),
                PACKAGE_PROFILE | LEGACY_PACKAGE_PROFILE
            )
        {
            return Err(invalid("unsupported QIF package profile"));
        }
        let canonical = serde_json::to_vec(&package).map_err(json_error)?;
        if canonical != package_bytes {
            return Err(invalid("QIF package manifest is not canonical JSON"));
        }
        package.model.validate()?;
        super::reader::validate_manifest_limits(&package.model)?;
        let mut unsigned = package.clone();
        unsigned.root_hash.clear();
        let unsigned_bytes = serde_json::to_vec(&unsigned).map_err(json_error)?;
        let metadata_bytes = read_bounded_file(
            &resolve_path(&root, &package.metadata_path)?,
            super::MAX_HEADER_SECTION,
            "QIF package metadata",
        )?;
        let index = read_bounded_file(
            &resolve_path(&root, &package.index_path)?,
            super::MAX_INDEX_BYTES,
            "QIF package index",
        )?;
        if crate::sha256_hex(&metadata_bytes) != package.metadata_sha256
            || crate::sha256_hex(&index) != package.index_sha256
        {
            return Err(invalid("QIF package control checksum mismatch"));
        }
        if package.root_hash
            != package_root_hash(&unsigned_bytes, &metadata_bytes, &index, &package.shards)
        {
            return Err(invalid("QIF package root hash mismatch"));
        }
        let metadata = super::metadata::decode(&metadata_bytes)?;
        let (tensors, chunks) = parse_package_index(&index, &package.model)?;
        let mut entries = BTreeMap::new();
        for entry in package.shards {
            let file = open_shard(&root, &entry.path)?;
            let size = file.metadata().map_err(io_error)?.len();
            if size != entry.size {
                return Err(invalid(format!("QIF shard {} size mismatch", entry.path)));
            }
            if entries.insert(entry.path.clone(), entry).is_some() {
                return Err(invalid("duplicate QIF package shard"));
            }
        }
        for chunk in &chunks {
            let path = format!("shard-{:05}.qifd", chunk.shard_id);
            let entry = entries
                .get(&path)
                .ok_or_else(|| invalid("QIF package index references a missing shard"))?;
            if chunk
                .stored_offset
                .checked_add(chunk.stored_length)
                .is_none()
                || chunk.stored_offset + chunk.stored_length > entry.size
                || chunk.stored_offset % ALIGNMENT as u64 != 0
            {
                return Err(invalid("package chunk exceeds its shard"));
            }
        }
        let mut occupied = BTreeMap::<String, Vec<(u64, u64)>>::new();
        for chunk in &chunks {
            let path = format!("shard-{:05}.qifd", chunk.shard_id);
            let end = chunk
                .stored_offset
                .checked_add(chunk.stored_length)
                .ok_or_else(|| invalid("package chunk range overflows"))?;
            occupied
                .entry(path)
                .or_default()
                .push((chunk.stored_offset, end));
        }
        for ranges in occupied.values_mut() {
            ranges.sort_unstable_by_key(|(start, _)| *start);
            for pair in ranges.windows(2) {
                if pair[0].1 > pair[1].0 {
                    return Err(invalid("package chunks overlap"));
                }
            }
        }
        let expected = package
            .model
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<BTreeSet<_>>();
        let actual = tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(invalid(
                "QIF package index tensor map disagrees with manifest",
            ));
        }
        let mut indexed_shards = BTreeMap::<String, BTreeSet<String>>::new();
        for tensor in &tensors {
            let mut physical = tensor;
            while let Some(alias) = physical.alias_of {
                physical = tensors
                    .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                    .ok_or_else(|| invalid("package alias target is out of range"))?;
            }
            for chunk in &chunks[physical.chunk_start..physical.chunk_start + physical.chunk_count]
            {
                indexed_shards
                    .entry(format!("shard-{:05}.qifd", chunk.shard_id))
                    .or_default()
                    .insert(tensor.name.clone());
            }
        }
        for (path, entry) in &entries {
            let listed = entry.tensors.iter().cloned().collect::<BTreeSet<_>>();
            if listed != indexed_shards.get(path).cloned().unwrap_or_default() {
                return Err(invalid(format!(
                    "package shard {path} tensor list disagrees with index"
                )));
            }
            if entry.rank == 0 || entry.coordinate.len() != entry.rank as usize {
                return Err(invalid(format!(
                    "package shard {path} mesh coordinate is invalid"
                )));
            }
        }
        let control_length = package_bytes.len() + metadata_bytes.len() + index.len();
        let artifact_length = entries
            .values()
            .try_fold(control_length as u64, |sum, entry| {
                sum.checked_add(entry.size)
                    .ok_or_else(|| invalid("package size overflows u64"))
            })?;
        Ok(Self {
            root,
            manifest: package.model,
            metadata,
            tensors,
            chunks,
            open_shard: None,
            entries,
            artifact_length,
            control_length,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| invalid(format!("unknown tensor {name}")))
    }
    pub fn tensor_shard(&self, name: &str) -> Result<&str> {
        let mut tensor = self.tensor_info(name)?;
        while let Some(alias) = tensor.alias_of {
            tensor = self
                .tensors
                .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("package alias target is out of range"))?;
        }
        let chunk = self.chunks[tensor.chunk_start].shard_id;
        self.entries
            .get_key_value(&format!("shard-{chunk:05}.qifd"))
            .map(|(path, _)| path.as_str())
            .ok_or_else(|| invalid("package shard is missing"))
    }
    pub fn read_range(&mut self, name: &str, range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        let mut tensor = self.tensor_info(name)?.clone();
        while let Some(alias) = tensor.alias_of {
            tensor = self
                .tensors
                .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("package alias target is out of range"))?
                .clone();
        }
        let raw_length = usize::try_from(tensor.raw_length)
            .map_err(|_| invalid("tensor length overflows usize"))?;
        if range.start > range.end || range.end > raw_length {
            return Err(invalid("tensor range exceeds the stored tensor"));
        }
        let mut output = super::zeroed_bytes(range.end - range.start)?;
        for index in tensor.chunk_start..tensor.chunk_start + tensor.chunk_count {
            let chunk = &self.chunks[index];
            let start = usize::try_from(chunk.raw_offset)
                .map_err(|_| invalid("chunk offset overflows usize"))?;
            let end = start
                .checked_add(
                    usize::try_from(chunk.raw_length)
                        .map_err(|_| invalid("chunk length overflows usize"))?,
                )
                .ok_or_else(|| invalid("chunk range overflows usize"))?;
            let copy_start = range.start.max(start);
            let copy_end = range.end.min(end);
            if copy_start >= copy_end {
                continue;
            }
            let shard_name = format!("shard-{:05}.qifd", chunk.shard_id);
            let file = shard_file(&self.root, &mut self.open_shard, &shard_name)?;
            if chunk.codec == CODEC_NONE {
                let offset = chunk
                    .stored_offset
                    .checked_add((copy_start - start) as u64)
                    .ok_or_else(|| invalid("package range offset overflows"))?;
                let output_start = copy_start - range.start;
                read_at(
                    file,
                    offset,
                    &mut output[output_start..output_start + copy_end - copy_start],
                )?;
                continue;
            }
            let mut stored = super::zeroed_bytes(
                usize::try_from(chunk.stored_length)
                    .map_err(|_| invalid("chunk length overflows usize"))?,
            )?;
            read_at(file, chunk.stored_offset, &mut stored)?;
            let decoded = decompress(chunk.codec, &stored, end - start)?;
            let source_start = copy_start - start;
            let source_end = copy_end - start;
            let output_start = copy_start - range.start;
            output[output_start..output_start + source_end - source_start]
                .copy_from_slice(&decoded[source_start..source_end]);
        }
        Ok(output)
    }
    pub fn verify(&mut self) -> Result<()> {
        for tensor in &self.tensors {
            if tensor.alias_of.is_some() {
                continue;
            }
            let mut verifier = PayloadVerifier::new(dtype_from_id(tensor.dtype)?, &tensor.name);
            for chunk in &self.chunks[tensor.chunk_start..tensor.chunk_start + tensor.chunk_count] {
                let path = format!("shard-{:05}.qifd", chunk.shard_id);
                let file = shard_file(&self.root, &mut self.open_shard, &path)?;
                chunk.verify_file(file, 0, &mut verifier)?;
            }
            verifier.finish(tensor.raw_length, &tensor.checksum_sha256)?;
        }
        for (path, entry) in &self.entries {
            let file = shard_file(&self.root, &mut self.open_shard, path)?;
            file.seek(SeekFrom::Start(0)).map_err(io_error)?;
            if crate::sha256_reader(file).map_err(io_error)? != entry.sha256 {
                return Err(invalid(format!("package shard {path} checksum mismatch")));
            }
        }
        Ok(())
    }
}

fn dtype_from_id(id: u32) -> Result<quantinfer_core::DType> {
    match id {
        1 => Ok(quantinfer_core::DType::F32),
        2 => Ok(quantinfer_core::DType::F16),
        3 => Ok(quantinfer_core::DType::BF16),
        4 => Ok(quantinfer_core::DType::Q4_0),
        5 => Ok(quantinfer_core::DType::Q4_K),
        6 => Ok(quantinfer_core::DType::Q5_K),
        7 => Ok(quantinfer_core::DType::Q6_K),
        8 => Ok(quantinfer_core::DType::Q8_0),
        value => Err(invalid(format!("unsupported tensor dtype id {value}"))),
    }
}

fn parse_package_index(
    bytes: &[u8],
    manifest: &ModelManifest,
) -> Result<(Vec<TensorInfo>, Vec<PackageChunk>)> {
    if bytes.len() < INDEX_HEADER_SIZE
        || &bytes[..4] != INDEX_MAGIC
        || u32_at(bytes, 4) != INDEX_VERSION
    {
        return Err(invalid("invalid package index header"));
    }
    let tensor_count =
        usize::try_from(u32_at(bytes, 8)).map_err(|_| invalid("tensor count overflows usize"))?;
    let chunk_count =
        usize::try_from(u32_at(bytes, 12)).map_err(|_| invalid("chunk count overflows usize"))?;
    if tensor_count > MAX_TENSORS
        || chunk_count > MAX_CHUNKS
        || tensor_count != manifest.tensors.len()
    {
        return Err(invalid("package index item count is invalid"));
    }
    let strings_len = usize::try_from(u64_at(bytes, 16))
        .map_err(|_| invalid("string table length overflows usize"))?;
    let shapes_len = usize::try_from(u64_at(bytes, 24))
        .map_err(|_| invalid("shape table length overflows usize"))?;
    let records_len =
        usize::try_from(u64_at(bytes, 32)).map_err(|_| invalid("record length overflows usize"))?;
    let chunks_len =
        usize::try_from(u64_at(bytes, 40)).map_err(|_| invalid("chunk length overflows usize"))?;
    if records_len != tensor_count * TENSOR_RECORD_SIZE
        || chunks_len != chunk_count * CHUNK_RECORD_SIZE
    {
        return Err(invalid("package index lengths are invalid"));
    }
    let strings_start = INDEX_HEADER_SIZE;
    let shapes_start = strings_start
        .checked_add(strings_len)
        .ok_or_else(|| invalid("package string table overflows"))?;
    let records_start = shapes_start
        .checked_add(shapes_len)
        .ok_or_else(|| invalid("package shape table overflows"))?;
    let chunks_start = records_start
        .checked_add(records_len)
        .ok_or_else(|| invalid("package tensor records overflow"))?;
    if chunks_start.checked_add(chunks_len) != Some(bytes.len()) || bytes.len() > MAX_INDEX_BYTES {
        return Err(invalid("package index bounds are invalid"));
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    for index in 0..chunk_count {
        let offset = chunks_start + index * CHUNK_RECORD_SIZE;
        let chunk = PackageChunk {
            shard_id: u32_at(bytes, offset),
            codec: u32_at(bytes, offset + 4),
            raw_offset: u64_at(bytes, offset + 8),
            raw_length: u64_at(bytes, offset + 16),
            stored_offset: u64_at(bytes, offset + 24),
            stored_length: u64_at(bytes, offset + 32),
            checksum_sha256: bytes[offset + 40..offset + 72].try_into().unwrap(),
        };
        if chunk.raw_length == 0 || chunk.stored_length == 0 {
            return Err(invalid("empty physical chunks are not allowed"));
        }
        super::codec::validate_lengths(chunk.codec, chunk.raw_length, chunk.stored_length)?;
        if !matches!(chunk.codec, CODEC_NONE | CODEC_ZSTD | CODEC_LZ4)
            || chunk.raw_offset.checked_add(chunk.raw_length).is_none()
            || chunk
                .stored_offset
                .checked_add(chunk.stored_length)
                .is_none()
            || (chunk.codec == CODEC_NONE && chunk.stored_length != chunk.raw_length)
        {
            return Err(invalid("package chunk range or codec is invalid"));
        }
        chunks.push(chunk);
    }
    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count);
    let by_name = manifest
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut names = BTreeSet::new();
    let mut referenced = vec![false; chunk_count];
    for index in 0..tensor_count {
        let offset = records_start + index * TENSOR_RECORD_SIZE;
        let tensor_id = u64_at(bytes, offset);
        if tensor_id != index as u64 {
            return Err(invalid("package tensor ids are not contiguous"));
        }
        let name = read_string_ref(bytes, strings_start, strings_len, u32_at(bytes, offset + 8))?;
        let manifest_tensor = by_name.get(name.as_str()).copied().ok_or_else(|| {
            invalid(format!(
                "package index tensor {tensor_id} is not in manifest"
            ))
        })?;
        if !names.insert(manifest_tensor.name.as_str()) {
            return Err(invalid("duplicate package tensor name"));
        }
        let shape = read_shape_ref(bytes, shapes_start, shapes_len, u32_at(bytes, offset + 12))?;
        let chunk_start = u32_at(bytes, offset + 40) as usize;
        let chunk_count_for_tensor = u32_at(bytes, offset + 44) as usize;
        let end_chunk = chunk_start
            .checked_add(chunk_count_for_tensor)
            .ok_or_else(|| invalid("package tensor chunks overflow"))?;
        if end_chunk > chunks.len() {
            return Err(invalid("package tensor chunks are invalid"));
        }
        if u32_at(bytes, offset + 36) != 0
            || u32_at(bytes, offset + 112) != 0
            || u32_at(bytes, offset + 116) != 0
        {
            return Err(invalid(
                "package tensor flags or reserved fields are not zero",
            ));
        }
        let alias_raw = u64_at(bytes, offset + 104);
        let alias_of = if alias_raw == ALIAS_NONE {
            None
        } else {
            if alias_raw >= index as u64 || chunk_count_for_tensor != 0 {
                return Err(invalid("package tensor alias target is invalid"));
            }
            Some(alias_raw)
        };
        let data_offset = u64_at(bytes, offset + 64);
        let tensor = TensorInfo {
            tensor_id,
            name,
            shape,
            dtype: u32_at(bytes, offset + 16),
            physical_encoding: u32_at(bytes, offset + 20),
            quantization_scheme: u32_at(bytes, offset + 24),
            layout: u32_at(bytes, offset + 28),
            codec: u32_at(bytes, offset + 32),
            raw_length: u64_at(bytes, offset + 48),
            stored_length: u64_at(bytes, offset + 56),
            checksum_sha256: bytes[offset + 72..offset + 104].try_into().unwrap(),
            alias_of,
            data_offset,
            chunk_start,
            chunk_count: chunk_count_for_tensor,
        };
        validate_tensor_descriptor(&tensor, manifest_tensor)?;
        if let Some(alias) = tensor.alias_of {
            let target = tensors
                .get(usize::try_from(alias).map_err(|_| invalid("alias id overflows usize"))?)
                .ok_or_else(|| invalid("package alias target is out of range"))?;
            if target.alias_of.is_some() || target.chunk_count == 0 {
                return Err(invalid("package alias target must be physical"));
            }
            if tensor.shape != target.shape
                || tensor.dtype != target.dtype
                || tensor.physical_encoding != target.physical_encoding
                || tensor.quantization_scheme != target.quantization_scheme
                || tensor.layout != target.layout
                || tensor.codec != target.codec
                || tensor.raw_length != target.raw_length
                || tensor.stored_length != target.stored_length
                || tensor.checksum_sha256 != target.checksum_sha256
                || data_offset != chunks[target.chunk_start].stored_offset
            {
                return Err(invalid("package alias metadata disagrees with target"));
            }
        } else {
            if chunk_count_for_tensor == 0 {
                return Err(invalid("non-alias package tensor has no chunks"));
            }
            for used in &mut referenced[chunk_start..end_chunk] {
                if std::mem::replace(used, true) {
                    return Err(invalid(
                        "package physical chunk is referenced by multiple tensors",
                    ));
                }
            }
            if tensor.codec != chunks[chunk_start].codec
                || chunks[chunk_start..end_chunk]
                    .iter()
                    .any(|chunk| chunk.codec != tensor.codec)
            {
                return Err(invalid("package tensor chunk metadata disagrees"));
            }
            let mut raw_sum = 0u64;
            let mut stored_sum = 0u64;
            let mut shard_end = BTreeMap::<u32, u64>::new();
            for (position, chunk) in chunks[chunk_start..end_chunk].iter().enumerate() {
                if position == 0 && chunk.stored_offset != data_offset {
                    return Err(invalid(
                        "package tensor data offset disagrees with first chunk",
                    ));
                }
                if let Some(previous_end) = shard_end.get(&chunk.shard_id) {
                    if chunk.stored_offset < *previous_end {
                        return Err(invalid("package tensor chunks overlap or are out of order"));
                    }
                }
                if chunk.raw_offset != raw_sum {
                    return Err(invalid("package tensor chunk offsets are not contiguous"));
                }
                raw_sum = raw_sum
                    .checked_add(chunk.raw_length)
                    .ok_or_else(|| invalid("package tensor raw length overflows"))?;
                stored_sum = stored_sum
                    .checked_add(chunk.stored_length)
                    .ok_or_else(|| invalid("package tensor stored length overflows"))?;
                shard_end.insert(
                    chunk.shard_id,
                    chunk
                        .stored_offset
                        .checked_add(chunk.stored_length)
                        .ok_or_else(|| invalid("package tensor stored offset overflows"))?,
                );
            }
            if raw_sum != tensor.raw_length || stored_sum != tensor.stored_length {
                return Err(invalid("package tensor lengths disagree with chunks"));
            }
        }
        tensors.push(tensor);
    }
    if referenced.contains(&false) {
        return Err(invalid("package index contains an unreferenced chunk"));
    }
    Ok((tensors, chunks))
}

fn encode_index(
    tensor_count: usize,
    chunk_count: usize,
    strings: &[u8],
    shapes: &[u8],
    records: &[u8],
    chunks: &[u8],
) -> Result<Vec<u8>> {
    let mut index = Vec::with_capacity(
        INDEX_HEADER_SIZE + strings.len() + shapes.len() + records.len() + chunks.len(),
    );
    index.extend_from_slice(INDEX_MAGIC);
    index.extend_from_slice(&INDEX_VERSION.to_le_bytes());
    index.extend_from_slice(&(tensor_count as u32).to_le_bytes());
    index.extend_from_slice(&(chunk_count as u32).to_le_bytes());
    index.extend_from_slice(&(strings.len() as u64).to_le_bytes());
    index.extend_from_slice(&(shapes.len() as u64).to_le_bytes());
    index.extend_from_slice(&(records.len() as u64).to_le_bytes());
    index.extend_from_slice(&(chunks.len() as u64).to_le_bytes());
    index.extend_from_slice(strings);
    index.extend_from_slice(shapes);
    index.extend_from_slice(records);
    index.extend_from_slice(chunks);
    if index.len() > MAX_INDEX_BYTES {
        return Err(invalid("package index exceeds the size limit"));
    }
    Ok(index)
}

#[allow(clippy::too_many_arguments)]
fn encode_tensor_record(
    id: u64,
    name: u32,
    shape: u32,
    dtype: u32,
    encoding: u32,
    scheme: u32,
    layout: u32,
    codec: u32,
    chunk_start: u32,
    chunk_count: u32,
    raw_len: usize,
    stored_len: usize,
    data_offset: usize,
    hash: [u8; HASH_SIZE],
    alias_target: u64,
) -> [u8; TENSOR_RECORD_SIZE] {
    let mut record = [0_u8; TENSOR_RECORD_SIZE];
    let mut cursor = 0usize;
    record[cursor..cursor + 8].copy_from_slice(&id.to_le_bytes());
    cursor += 8;
    for value in [
        name,
        shape,
        dtype,
        encoding,
        scheme,
        layout,
        codec,
        0,
        chunk_start,
        chunk_count,
    ] {
        record[cursor..cursor + 4].copy_from_slice(&value.to_le_bytes());
        cursor += 4;
    }
    for value in [raw_len as u64, stored_len as u64, data_offset as u64] {
        record[cursor..cursor + 8].copy_from_slice(&value.to_le_bytes());
        cursor += 8;
    }
    record[cursor..cursor + HASH_SIZE].copy_from_slice(&hash);
    cursor += HASH_SIZE;
    record[cursor..cursor + 8].copy_from_slice(&alias_target.to_le_bytes());
    record
}

fn encode_chunk(
    shard: u32,
    codec: u32,
    raw_offset: usize,
    raw_len: usize,
    stored_offset: usize,
    stored_len: usize,
    hash: [u8; HASH_SIZE],
) -> [u8; CHUNK_RECORD_SIZE] {
    let mut record = [0_u8; CHUNK_RECORD_SIZE];
    record[0..4].copy_from_slice(&shard.to_le_bytes());
    record[4..8].copy_from_slice(&codec.to_le_bytes());
    record[8..16].copy_from_slice(&(raw_offset as u64).to_le_bytes());
    record[16..24].copy_from_slice(&(raw_len as u64).to_le_bytes());
    record[24..32].copy_from_slice(&(stored_offset as u64).to_le_bytes());
    record[32..40].copy_from_slice(&(stored_len as u64).to_le_bytes());
    record[40..72].copy_from_slice(&hash);
    record
}

fn descriptor_ids(
    descriptor: &quantinfer_core::TensorStorageDescriptor,
) -> Result<(u32, u32, u32)> {
    super::encoding::validate_descriptor(descriptor)?;
    let dtype = match descriptor.dtype {
        quantinfer_core::DType::F32 => 1,
        quantinfer_core::DType::F16 => 2,
        quantinfer_core::DType::BF16 => 3,
        quantinfer_core::DType::Q4_0 => 4,
        quantinfer_core::DType::Q4_K => 5,
        quantinfer_core::DType::Q5_K => 6,
        quantinfer_core::DType::Q6_K => 7,
        quantinfer_core::DType::Q8_0 => 8,
    };
    let scheme = match descriptor.quantization {
        quantinfer_core::QuantizationScheme::None => 0,
        quantinfer_core::QuantizationScheme::Q8_0 => 1,
        quantinfer_core::QuantizationScheme::Q4_0 => 2,
        quantinfer_core::QuantizationScheme::Q4_K => 3,
        quantinfer_core::QuantizationScheme::Q5_K => 4,
        quantinfer_core::QuantizationScheme::Q6_K => 5,
        quantinfer_core::QuantizationScheme::Mixed => 6,
    };
    let layout = match descriptor.physical_layout.as_str() {
        "row_major" => 1,
        "output_rows" => 2,
        crate::GGML_OUTPUT_ROWS_LAYOUT => 2,
        value => return Err(invalid(format!("unsupported tensor layout {value}"))),
    };
    Ok((dtype, scheme, layout))
}

fn package_payload_size(descriptor: &quantinfer_core::TensorStorageDescriptor) -> Result<usize> {
    super::encoding::validate_descriptor(descriptor)?;
    let dtype = match descriptor.dtype {
        quantinfer_core::DType::F32 => crate::GgmlType::F32,
        quantinfer_core::DType::F16 => crate::GgmlType::F16,
        quantinfer_core::DType::BF16 => crate::GgmlType::BF16,
        quantinfer_core::DType::Q8_0 => crate::GgmlType::Q8_0,
        quantinfer_core::DType::Q4_0 => crate::GgmlType::Q4_0,
        quantinfer_core::DType::Q4_K => crate::GgmlType::Q4_K,
        quantinfer_core::DType::Q5_K => crate::GgmlType::Q5_K,
        quantinfer_core::DType::Q6_K => crate::GgmlType::Q6_K,
    };
    crate::tensor_byte_size(dtype, descriptor.logical_shape.dims())
}

fn package_root_hash(
    unsigned: &[u8],
    metadata: &[u8],
    index: &[u8],
    shards: &[ShardManifest],
) -> String {
    let mut digest = Sha256::new();
    digest.update(unsigned);
    digest.update(metadata);
    digest.update(index);
    for shard in shards {
        digest.update(shard.sha256.as_bytes());
    }
    let bytes: [u8; HASH_SIZE] = digest.finalize().into();
    hex_encode(&bytes)
}

fn resolve_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || !path
            .components()
            .any(|component| matches!(component, Component::Normal(_)))
    {
        return Err(invalid(format!("invalid package path {relative}")));
    }
    let path = fs::canonicalize(root.join(path)).map_err(io_error)?;
    if !path.starts_with(root) {
        return Err(invalid(format!("package path escapes root: {relative}")));
    }
    Ok(path)
}

fn open_shard(root: &Path, name: &str) -> Result<File> {
    let path = resolve_path(root, name)?;
    if !fs::metadata(&path).map_err(io_error)?.is_file() {
        return Err(invalid(format!("QIF shard {name} is not a regular file")));
    }
    crate::manifest::open_readonly_nofollow(&path).map_err(io_error)
}

fn shard_file<'a>(
    root: &Path,
    open: &'a mut Option<OpenShard>,
    name: &str,
) -> Result<&'a mut File> {
    if open.as_ref().is_none_or(|shard| shard.name != name) {
        // Release the current descriptor before opening another shard.
        *open = None;
        *open = Some(OpenShard {
            name: name.to_owned(),
            file: open_shard(root, name)?,
        });
    }
    Ok(&mut open.as_mut().expect("shard is open").file)
}

fn write_atomic_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    drop(file);
    fs::rename(temporary, path).map_err(io_error)
}

fn read_bounded_file(path: &Path, maximum: usize, label: &str) -> Result<Vec<u8>> {
    let size = fs::metadata(path).map_err(io_error)?.len();
    if size > maximum as u64 {
        return Err(invalid(format!("{label} exceeds the size limit")));
    }
    let bytes = fs::read(path).map_err(io_error)?;
    if bytes.len() > maximum {
        return Err(invalid(format!("{label} exceeds the size limit")));
    }
    Ok(bytes)
}

fn read_at(file: &mut File, offset: u64, bytes: &mut [u8]) -> Result<()> {
    file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
    file.read_exact(bytes).map_err(io_error)
}
fn io_error(error: std::io::Error) -> FormatError {
    FormatError::Io(format!("QIF package I/O error: {error}"))
}
fn json_error(error: impl std::fmt::Display) -> FormatError {
    FormatError::Invalid(format!("invalid QIF package manifest: {error}"))
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
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

#[cfg(test)]
mod tests;
