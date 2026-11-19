use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::{self, Read, Seek, Write};

use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};

use quantinfer_core::{
    DType, QuantizationScheme, Shape, TensorStorageDescriptor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
};

use crate::gguf::{GgufReader, SpoolSource};
use crate::{GgmlType, GgufTensorInfo, GgufValue, ModelManifest, Result, TensorManifest};

use super::metadata::MetadataValue;
use super::package::write_package_from_source;
use super::payload::validate_payload;
use super::{
    align, canonicalize_manifest_layouts, invalid, metadata_from_gguf, ALIAS_NONE, ALIGNMENT,
    CODEC_LZ4, CODEC_NONE, CODEC_ZSTD, HASH_SHA256, HASH_SIZE, HEADER_SIZE, INDEX_MAGIC,
    INDEX_VERSION, MAGIC, MAX_CHUNKS, MAX_INDEX_BYTES, MAX_RANK, MAX_TENSORS, VERSION,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    None,
    Zstd,
    Lz4,
}

trait Scratch: Read + Write + Seek {}
impl<T: Read + Write + Seek> Scratch for T {}

enum StagedPayload<'a> {
    Memory(Vec<(usize, Vec<u8>)>),
    Scratch {
        file: &'a mut dyn Scratch,
        written: usize,
    },
}

impl StagedPayload<'_> {
    fn append(&mut self, offset: usize, bytes: Cow<'_, [u8]>) -> Result<()> {
        match self {
            Self::Memory(chunks) => chunks.push((offset, bytes.into_owned())),
            Self::Scratch { file, written } => {
                write_padding(file, *written, offset)?;
                file.write_all(&bytes).map_err(write_error)?;
                *written = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| invalid("payload length overflows"))?;
            }
        }
        Ok(())
    }

    fn publish(self, output: &mut impl Write, length: usize) -> Result<()> {
        match self {
            Self::Memory(chunks) => {
                let mut written = 0;
                for (offset, bytes) in chunks {
                    write_padding(output, written, offset)?;
                    output.write_all(&bytes).map_err(write_error)?;
                    written = offset
                        .checked_add(bytes.len())
                        .ok_or_else(|| invalid("payload length overflows"))?;
                }
                write_padding(output, written, length)
            }
            Self::Scratch { mut file, written } => {
                write_padding(&mut file, written, length)?;
                file.rewind().map_err(write_error)?;
                let length =
                    u64::try_from(length).map_err(|_| invalid("payload length overflows u64"))?;
                let copied = io::copy(&mut file.take(length), output).map_err(write_error)?;
                if copied != length {
                    return Err(invalid("QIF scratch payload is truncated"));
                }
                Ok(())
            }
        }
    }
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
}

impl Codec {
    pub const fn id(self) -> u32 {
        match self {
            Self::None => CODEC_NONE,
            Self::Zstd => CODEC_ZSTD,
            Self::Lz4 => CODEC_LZ4,
        }
    }
}

/// Transcode a supported GGUF model to the new QIF wire format without
/// dequantizing or requantizing its tensor bytes.
pub fn import_gguf(
    reader: &GgufReader,
    model_id: impl Into<String>,
    output: &mut impl Write,
) -> Result<()> {
    import_gguf_with_codec(reader, model_id, output, Codec::None)
}

/// Transcode GGUF to SINGLE with independently compressed chunks.
pub fn import_gguf_with_codec(
    reader: &GgufReader,
    model_id: impl Into<String>,
    output: &mut impl Write,
    codec: Codec,
) -> Result<()> {
    let (manifest, metadata, payloads) = import_parts(reader, model_id)?;
    write_qif_v2_with_codec(
        manifest,
        &metadata,
        |name| Ok(payloads[name].as_ref()),
        output,
        codec,
    )
}

/// Transcode GGUF to a native QIF-PACKAGE directory without requantization.
pub fn import_gguf_package(
    reader: &GgufReader,
    model_id: impl Into<String>,
    output: impl AsRef<std::path::Path>,
    max_shard_bytes: Option<usize>,
    codec: Codec,
) -> Result<()> {
    let (manifest, metadata, payloads) = import_parts(reader, model_id)?;
    super::write_qif_v2_package_with_codec(
        manifest,
        &metadata,
        |name| Ok(payloads[name].as_ref()),
        output,
        max_shard_bytes,
        codec,
    )
}

pub(crate) fn write_spooled_single<F: Read + Seek>(
    mut source: SpoolSource<F>,
    model_id: impl Into<String>,
    output: &mut impl Write,
    scratch: &mut (impl Read + Write + Seek),
    codec: Codec,
) -> Result<()> {
    let (manifest, metadata) = describe_gguf_controls(
        &source.controls.metadata,
        &source.controls.tensors,
        model_id,
        source.ggml_rows()?,
    )?;
    let legacy = source.legacy_blocks()?;
    write_single_from_source(
        manifest,
        &metadata,
        |name| spooled_tensor(&mut source, legacy, name),
        output,
        codec,
        scratch,
    )
}

pub(crate) fn write_spooled_package<F: Read + Seek>(
    mut source: SpoolSource<F>,
    model_id: impl Into<String>,
    output: &std::path::Path,
    max_shard_bytes: Option<usize>,
    codec: Codec,
) -> Result<()> {
    let (manifest, metadata) = describe_gguf_controls(
        &source.controls.metadata,
        &source.controls.tensors,
        model_id,
        source.ggml_rows()?,
    )?;
    let legacy = source.legacy_blocks()?;
    write_package_from_source(
        manifest,
        &metadata,
        |name| spooled_tensor(&mut source, legacy, name),
        output,
        max_shard_bytes,
        codec,
    )
}

fn spooled_tensor<'a>(
    source: &mut SpoolSource<impl Read + Seek>,
    legacy: bool,
    name: &str,
) -> Result<Cow<'a, [u8]>> {
    let (dtype, mut bytes) = source.read_tensor(name)?;
    if dtype == GgmlType::Q4_0 && legacy {
        bytes = repack_q4_0(&bytes);
    }
    Ok(Cow::Owned(bytes))
}

type Imported<'a> = (
    ModelManifest,
    BTreeMap<String, MetadataValue>,
    BTreeMap<String, Cow<'a, [u8]>>,
);
fn import_parts(reader: &GgufReader, model_id: impl Into<String>) -> Result<Imported<'_>> {
    let (manifest, metadata) = describe_gguf(reader, model_id)?;
    let mut payloads = BTreeMap::<String, Cow<'_, [u8]>>::new();
    for tensor in reader.tensors() {
        let bytes = reader.tensor_bytes(&tensor.name)?;
        let payload =
            if tensor.dtype == GgmlType::Q4_0 && reader.uses_legacy_quant_block_layout()? {
                Cow::Owned(repack_q4_0(bytes))
            } else {
                Cow::Borrowed(bytes)
            };
        payloads.insert(tensor.name.clone(), payload);
    }
    Ok((manifest, metadata, payloads))
}

/// Describe GGUF controls in the native QIF coordinate system without reading
/// or copying tensor payloads. Checksums are filled when the artifact is
/// written. Native QIF callers should retain its embedded manifest/metadata.
pub fn describe_gguf(
    reader: &GgufReader,
    model_id: impl Into<String>,
) -> Result<(ModelManifest, BTreeMap<String, MetadataValue>)> {
    if reader.embedded_manifest().is_some() {
        return Err(invalid(
            "source must be GGUF; repack native QIF through its native reader/writer",
        ));
    }
    describe_gguf_controls(
        reader.metadata(),
        reader.tensors(),
        model_id,
        reader.uses_ggml_output_rows()?,
    )
}

fn describe_gguf_controls(
    metadata: &BTreeMap<String, GgufValue>,
    tensors: &[GgufTensorInfo],
    model_id: impl Into<String>,
    ggml: bool,
) -> Result<(ModelManifest, BTreeMap<String, MetadataValue>)> {
    let mut metadata = metadata.clone();
    let architecture = match metadata.get("general.architecture") {
        Some(GgufValue::String(architecture)) => architecture.as_str(),
        None => return Err(invalid("source GGUF is missing general.architecture")),
        Some(_) => return Err(invalid("source architecture must be a string")),
    };
    let mut manifest = ModelManifest::new(model_id, architecture);
    metadata.insert(
        "quantinfer.qif.ggml_dimension_order".to_owned(),
        GgufValue::Bool(ggml),
    );
    let mut schemes = Vec::new();
    for tensor in tensors {
        let (dtype, scheme, block_size) = descriptor_parts(tensor.dtype)?;
        let mut shape = tensor.shape.clone();
        if ggml {
            shape.reverse();
        }
        let shape = Shape::new(shape)?;
        let mut descriptor = if scheme == QuantizationScheme::None {
            TensorStorageDescriptor::floating(dtype, shape)?
        } else {
            if !schemes.contains(&scheme) {
                schemes.push(scheme);
            }
            TensorStorageDescriptor::quantized(dtype, scheme, shape, block_size)?
        };
        if ggml && tensor.shape.len() == 2 && !crate::is_embedding_storage_name(&tensor.name) {
            descriptor.physical_layout = crate::GGML_OUTPUT_ROWS_LAYOUT.to_owned();
            if scheme != QuantizationScheme::None {
                descriptor.quantization_axis = quantinfer_core::QuantizationAxis::Axis(0);
            }
        }
        manifest.tensors.push(TensorManifest {
            name: tensor.name.clone(),
            descriptor,
            checksum_sha256: None,
        });
    }
    manifest.quantization = match metadata.get("quantinfer.quantization.variant") {
        Some(GgufValue::String(variant)) if variant == "mixed" => QuantizationScheme::Mixed,
        _ => match schemes.as_slice() {
            [] => QuantizationScheme::None,
            [scheme] => *scheme,
            _ => QuantizationScheme::Mixed,
        },
    };
    manifest.validate()?;
    Ok((manifest, metadata_from_gguf(&metadata)))
}

/// Write a QIF-SINGLE artifact using the required uncompressed codec.
pub fn write_qif_v2<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
) -> Result<()> {
    write_qif_v2_with_codec(manifest, metadata, tensor_bytes, output, Codec::None)
}

/// Write a QIF artifact using independent compressed chunks of at most 4 MiB.
/// Uncompressed tensors retain a single directly addressable chunk.
/// The default `write_qif_v2` uses the uncompressed profile, whose reader can
/// return borrowed tensor views.
pub fn write_qif_v2_with_codec<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
    codec: Codec,
) -> Result<()> {
    write_borrowed(manifest, metadata, tensor_bytes, output, codec, None)
}

pub fn write_qif_v2_signed<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
    codec: Codec,
    signing_key: &SigningKey,
) -> Result<()> {
    write_borrowed(
        manifest,
        metadata,
        tensor_bytes,
        output,
        codec,
        Some(signing_key),
    )
}

fn write_borrowed<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
    codec: Codec,
    signing_key: Option<&SigningKey>,
) -> Result<()> {
    let mut loaded = BTreeMap::new();
    write_qif_v2_inner(
        manifest,
        metadata,
        |name| {
            if let Some(&bytes) = loaded.get(name) {
                return Ok(Cow::Borrowed(bytes));
            }
            let bytes = tensor_bytes(name)?;
            loaded.insert(name.to_owned(), bytes);
            Ok(Cow::Borrowed(bytes))
        },
        output,
        codec,
        signing_key,
        StagedPayload::Memory(Vec::new()),
    )
}

fn write_single_from_source<'a>(
    manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    tensor_bytes: impl FnMut(&str) -> Result<Cow<'a, [u8]>>,
    output: &mut impl Write,
    codec: Codec,
    scratch: &mut (impl Read + Write + Seek),
) -> Result<()> {
    scratch.rewind().map_err(write_error)?;
    write_qif_v2_inner(
        manifest,
        metadata,
        tensor_bytes,
        output,
        codec,
        None,
        StagedPayload::Scratch {
            file: scratch,
            written: 0,
        },
    )
}

fn write_qif_v2_inner<'a>(
    mut manifest: ModelManifest,
    metadata: &BTreeMap<String, MetadataValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<Cow<'a, [u8]>>,
    output: &mut impl Write,
    codec: Codec,
    signing_key: Option<&SigningKey>,
    mut staging: StagedPayload<'_>,
) -> Result<()> {
    if manifest.tensors.len() > MAX_TENSORS {
        return Err(invalid("tensor count exceeds the limit"));
    }
    canonicalize_manifest_layouts(&mut manifest);
    manifest.validate()?;
    super::reader::validate_manifest_limits(&manifest)?;
    let metadata_bytes = super::metadata::encode(metadata)?;

    let (index, payload_length) =
        build_index(&mut manifest, codec.id(), &mut tensor_bytes, &mut staging)?;
    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| invalid(format!("cannot serialize manifest: {error}")))?;
    if manifest_bytes.len() > 64 * 1024 * 1024 {
        return Err(invalid("manifest exceeds the size limit"));
    }
    let root_hash = root_hash(&manifest_bytes, &metadata_bytes, &index);

    let manifest_offset = align(HEADER_SIZE)?;
    let metadata_offset = align(
        manifest_offset
            .checked_add(manifest_bytes.len())
            .ok_or_else(|| invalid("manifest range overflows"))?,
    )?;
    let index_offset = align(
        metadata_offset
            .checked_add(metadata_bytes.len())
            .ok_or_else(|| invalid("metadata range overflows"))?,
    )?;
    let index_end = align(
        index_offset
            .checked_add(index.len())
            .ok_or_else(|| invalid("index range overflows"))?,
    )?;
    let signature = signing_key.map(|key| super::signature::encode(&root_hash, key));
    let signature_offset = if signature.is_some() { index_end } else { 0 };
    let payload_offset = align(if let Some(signature) = &signature {
        signature_offset
            .checked_add(signature.len())
            .ok_or_else(|| invalid("signature range overflows"))?
    } else {
        index_end
    })?;
    let header = encode_header(
        manifest_offset,
        manifest_bytes.len(),
        metadata_offset,
        metadata_bytes.len(),
        index_offset,
        index.len(),
        payload_offset,
        payload_length,
        root_hash,
        signature_offset,
        signature.as_ref().map_or(0, |value| value.len()),
    )?;
    output.write_all(&header).map_err(write_error)?;
    write_padding(output, HEADER_SIZE, manifest_offset)?;
    output.write_all(&manifest_bytes).map_err(write_error)?;
    write_padding(
        output,
        manifest_offset + manifest_bytes.len(),
        metadata_offset,
    )?;
    output.write_all(&metadata_bytes).map_err(write_error)?;
    write_padding(output, metadata_offset + metadata_bytes.len(), index_offset)?;
    output.write_all(&index).map_err(write_error)?;
    write_padding(
        output,
        index_offset + index.len(),
        if signature.is_some() {
            signature_offset
        } else {
            payload_offset
        },
    )?;
    if let Some(signature) = signature {
        output.write_all(&signature).map_err(write_error)?;
        write_padding(output, signature_offset + signature.len(), payload_offset)?;
    }
    staging.publish(output, payload_length)
}

#[allow(clippy::too_many_arguments)]
fn encode_header(
    manifest_offset: usize,
    manifest_length: usize,
    metadata_offset: usize,
    metadata_length: usize,
    index_offset: usize,
    index_length: usize,
    payload_offset: usize,
    payload_length: usize,
    root_hash: [u8; HASH_SIZE],
    signature_offset: usize,
    signature_length: usize,
) -> Result<Vec<u8>> {
    let mut header = Vec::with_capacity(HEADER_SIZE);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&0_u32.to_le_bytes());
    header.extend_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
    header.extend_from_slice(&HASH_SHA256.to_le_bytes());
    header.extend_from_slice(&INDEX_VERSION.to_le_bytes());
    for value in [
        manifest_offset,
        manifest_length,
        metadata_offset,
        metadata_length,
        index_offset,
        index_length,
        payload_offset,
        payload_length,
    ] {
        header.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| invalid("header value overflows u64"))?
                .to_le_bytes(),
        );
    }
    header.extend_from_slice(&root_hash);
    header.extend_from_slice(
        &u64::try_from(signature_offset)
            .map_err(|_| invalid("signature offset overflows u64"))?
            .to_le_bytes(),
    );
    header.extend_from_slice(
        &u64::try_from(signature_length)
            .map_err(|_| invalid("signature length overflows u64"))?
            .to_le_bytes(),
    );
    header.extend_from_slice(&0_u64.to_le_bytes());
    if header.len() != HEADER_SIZE {
        return Err(invalid("internal header size mismatch"));
    }
    Ok(header)
}

fn build_index<'a>(
    manifest: &mut ModelManifest,
    codec: u32,
    tensor_bytes: &mut impl FnMut(&str) -> Result<Cow<'a, [u8]>>,
    staging: &mut StagedPayload<'_>,
) -> Result<(Vec<u8>, usize)> {
    let mut strings = Vec::new();
    let mut shapes = Vec::new();
    let mut records = Vec::new();
    let mut chunks = Vec::new();
    let mut dedup = BTreeMap::<[u8; HASH_SIZE], Vec<usize>>::new();
    let mut physical = Vec::<Option<PhysicalTensor>>::with_capacity(manifest.tensors.len());
    let mut payload_offset = 0usize;
    for index in 0..manifest.tensors.len() {
        let tensor = &mut manifest.tensors[index];
        let payload = tensor_bytes(&tensor.name)?;
        let expected = payload_size(&tensor.descriptor)?;
        if payload.len() != expected {
            return Err(invalid(format!(
                "tensor {} has {} bytes, expected {expected}",
                tensor.name,
                payload.len()
            )));
        }
        validate_payload(tensor.descriptor.dtype, &payload, &tensor.name)?;
        let hash: [u8; HASH_SIZE] = Sha256::digest(&payload).into();
        tensor.checksum_sha256 = Some(crate::manifest::hex_encode(&hash));
        let tensor = &manifest.tensors[index];
        let name_offset = strings.len();
        strings.extend_from_slice(
            &u32::try_from(tensor.name.len())
                .map_err(|_| invalid("tensor name is too large"))?
                .to_le_bytes(),
        );
        strings.extend_from_slice(tensor.name.as_bytes());
        let shape_offset = shapes.len();
        let dims = tensor.descriptor.logical_shape.dims();
        if dims.len() > MAX_RANK {
            return Err(invalid("tensor rank exceeds the limit"));
        }
        shapes.extend_from_slice(&(u32::try_from(dims.len()).unwrap()).to_le_bytes());
        for dim in dims {
            shapes.extend_from_slice(
                &u64::try_from(*dim)
                    .map_err(|_| invalid("tensor dimension overflows u64"))?
                    .to_le_bytes(),
            );
        }
        let (dtype, encoding, scheme, layout) = descriptor_ids(&tensor.descriptor)?;
        let mut alias_target = None;
        for &candidate in dedup.get(&hash).into_iter().flatten() {
            let prior = &manifest.tensors[candidate];
            if prior.descriptor == tensor.descriptor && tensor_bytes(&prior.name)? == payload {
                alias_target = Some(candidate);
                break;
            }
        }
        let tensor = if let Some(target) = alias_target {
            let prior = physical[target].expect("alias target is physical");
            let record = encode_tensor_record(
                index as u64,
                u32::try_from(name_offset)
                    .map_err(|_| invalid("name table offset overflows u32"))?,
                u32::try_from(shape_offset)
                    .map_err(|_| invalid("shape table offset overflows u32"))?,
                dtype,
                encoding,
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
            )?;
            physical.push(None);
            record
        } else {
            let chunk_start = chunks.len() / 72;
            let chunk_bytes = if codec == CODEC_NONE {
                payload.len().max(1)
            } else {
                super::codec::DEFAULT_CHUNK_BYTES
            };
            let first_offset = payload_offset;
            let mut raw_offset = 0usize;
            let mut stored_length = 0usize;
            for raw in payload.chunks(chunk_bytes) {
                if chunks.len() / 72 >= MAX_CHUNKS {
                    return Err(invalid("chunk count exceeds the limit"));
                }
                let stored = if codec == CODEC_NONE {
                    Cow::Borrowed(raw)
                } else {
                    Cow::Owned(super::codec::compress(codec, raw)?)
                };
                let stored_hash: [u8; HASH_SIZE] = Sha256::digest(&stored).into();
                chunks.extend_from_slice(&encode_chunk(
                    0,
                    codec,
                    raw_offset,
                    raw.len(),
                    payload_offset,
                    stored.len(),
                    stored_hash,
                )?);
                let offset = payload_offset;
                stored_length = stored_length
                    .checked_add(stored.len())
                    .ok_or_else(|| invalid("stored tensor length overflows"))?;
                payload_offset = align(
                    payload_offset
                        .checked_add(stored.len())
                        .ok_or_else(|| invalid("payload offset overflows"))?,
                )?;
                staging.append(offset, stored)?;
                raw_offset += raw.len();
            }
            let info = PhysicalTensor {
                chunk_start: u32::try_from(chunk_start)
                    .map_err(|_| invalid("chunk index overflows u32"))?,
                chunk_count: u32::try_from(chunks.len() / 72 - chunk_start)
                    .map_err(|_| invalid("chunk count overflows u32"))?,
                raw_length: payload.len(),
                stored_length,
                data_offset: first_offset,
                hash,
                codec,
            };
            physical.push(Some(info));
            dedup.entry(hash).or_default().push(index);
            encode_tensor_record(
                index as u64,
                u32::try_from(name_offset)
                    .map_err(|_| invalid("name table offset overflows u32"))?,
                u32::try_from(shape_offset)
                    .map_err(|_| invalid("shape table offset overflows u32"))?,
                dtype,
                encoding,
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
            )?
        };
        records.extend_from_slice(&tensor);
    }
    if manifest.tensors.len() > MAX_TENSORS || chunks.len() / 72 > MAX_CHUNKS {
        return Err(invalid("index item limit exceeded"));
    }
    let mut index = Vec::new();
    index.extend_from_slice(INDEX_MAGIC);
    index.extend_from_slice(&INDEX_VERSION.to_le_bytes());
    index.extend_from_slice(&(manifest.tensors.len() as u32).to_le_bytes());
    index.extend_from_slice(&((chunks.len() / 72) as u32).to_le_bytes());
    index.extend_from_slice(&(strings.len() as u64).to_le_bytes());
    index.extend_from_slice(&(shapes.len() as u64).to_le_bytes());
    index.extend_from_slice(&(records.len() as u64).to_le_bytes());
    index.extend_from_slice(&(chunks.len() as u64).to_le_bytes());
    index.extend_from_slice(&strings);
    index.extend_from_slice(&shapes);
    index.extend_from_slice(&records);
    index.extend_from_slice(&chunks);
    if index.len() > MAX_INDEX_BYTES {
        return Err(invalid("index exceeds the size limit"));
    }
    Ok((index, payload_offset))
}

#[allow(clippy::too_many_arguments)]
fn encode_tensor_record(
    tensor_id: u64,
    name_offset: u32,
    shape_offset: u32,
    dtype: u32,
    encoding: u32,
    scheme: u32,
    layout: u32,
    codec: u32,
    chunk_start: u32,
    chunk_count: u32,
    raw_length: usize,
    stored_length: usize,
    data_offset: usize,
    hash: [u8; HASH_SIZE],
    alias_target: u64,
) -> Result<[u8; 120]> {
    let mut record = [0_u8; 120];
    let mut offset = 0;
    write_u64(&mut record, &mut offset, tensor_id);
    for value in [
        name_offset,
        shape_offset,
        dtype,
        encoding,
        scheme,
        layout,
        codec,
        0,
        chunk_start,
        chunk_count,
    ] {
        write_u32(&mut record, &mut offset, value);
    }
    write_u64(&mut record, &mut offset, raw_length as u64);
    write_u64(&mut record, &mut offset, stored_length as u64);
    write_u64(&mut record, &mut offset, data_offset as u64);
    record[offset..offset + HASH_SIZE].copy_from_slice(&hash);
    offset += HASH_SIZE;
    write_u64(&mut record, &mut offset, alias_target);
    write_u32(&mut record, &mut offset, 0);
    write_u32(&mut record, &mut offset, 0);
    if offset != record.len() {
        return Err(invalid("internal tensor record size mismatch"));
    }
    Ok(record)
}

fn encode_chunk(
    shard_id: u32,
    codec: u32,
    raw_offset: usize,
    raw_length: usize,
    stored_offset: usize,
    stored_length: usize,
    hash: [u8; HASH_SIZE],
) -> Result<[u8; 72]> {
    let mut record = [0_u8; 72];
    record[0..4].copy_from_slice(&shard_id.to_le_bytes());
    record[4..8].copy_from_slice(&codec.to_le_bytes());
    record[8..16].copy_from_slice(&(raw_offset as u64).to_le_bytes());
    record[16..24].copy_from_slice(&(raw_length as u64).to_le_bytes());
    record[24..32].copy_from_slice(&(stored_offset as u64).to_le_bytes());
    record[32..40].copy_from_slice(&(stored_length as u64).to_le_bytes());
    record[40..72].copy_from_slice(&hash);
    Ok(record)
}

fn write_u32(target: &mut [u8], offset: &mut usize, value: u32) {
    target[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
    *offset += 4;
}

fn write_u64(target: &mut [u8], offset: &mut usize, value: u64) {
    target[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
    *offset += 8;
}

fn root_hash(manifest: &[u8], metadata: &[u8], index: &[u8]) -> [u8; HASH_SIZE] {
    let mut digest = Sha256::new();
    digest.update(manifest);
    digest.update(metadata);
    digest.update(index);
    digest.finalize().into()
}

fn write_padding(output: &mut impl Write, current: usize, target: usize) -> Result<()> {
    if target < current {
        return Err(invalid("output offsets are not monotonic"));
    }
    let padding = target - current;
    let zeros = [0_u8; ALIGNMENT];
    let mut remaining = padding;
    while remaining > 0 {
        let count = remaining.min(zeros.len());
        output.write_all(&zeros[..count]).map_err(write_error)?;
        remaining -= count;
    }
    Ok(())
}

fn write_error(error: std::io::Error) -> crate::FormatError {
    crate::FormatError::Io(format!("cannot write QIF: {error}"))
}

fn payload_size(descriptor: &TensorStorageDescriptor) -> Result<usize> {
    super::encoding::validate_descriptor(descriptor)?;
    let dtype = match descriptor.dtype {
        DType::F32 => GgmlType::F32,
        DType::F16 => GgmlType::F16,
        DType::BF16 => GgmlType::BF16,
        DType::Q8_0 => GgmlType::Q8_0,
        DType::Q4_0 => GgmlType::Q4_0,
        DType::Q4_K => GgmlType::Q4_K,
        DType::Q5_K => GgmlType::Q5_K,
        DType::Q6_K => GgmlType::Q6_K,
    };
    crate::tensor_byte_size(dtype, descriptor.logical_shape.dims())
}

fn descriptor_parts(dtype: GgmlType) -> Result<(DType, QuantizationScheme, usize)> {
    Ok(match dtype {
        GgmlType::F32 => (DType::F32, QuantizationScheme::None, 0),
        GgmlType::F16 => (DType::F16, QuantizationScheme::None, 0),
        GgmlType::BF16 => (DType::BF16, QuantizationScheme::None, 0),
        GgmlType::Q8_0 => (DType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
        GgmlType::Q4_0 => (DType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
        GgmlType::Q4_K => (DType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
        GgmlType::Q5_K => (DType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
        GgmlType::Q6_K => (DType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
    })
}

fn descriptor_ids(descriptor: &TensorStorageDescriptor) -> Result<(u32, u32, u32, u32)> {
    super::encoding::validate_descriptor(descriptor)?;
    let dtype = match descriptor.dtype {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::Q8_0 => 8,
        DType::Q4_0 => 4,
        DType::Q4_K => 5,
        DType::Q5_K => 6,
        DType::Q6_K => 7,
    };
    let scheme = match descriptor.quantization {
        QuantizationScheme::None => 0,
        QuantizationScheme::Q8_0 => 1,
        QuantizationScheme::Q4_0 => 2,
        QuantizationScheme::Q4_K => 3,
        QuantizationScheme::Q5_K => 4,
        QuantizationScheme::Q6_K => 5,
        QuantizationScheme::Mixed => 6,
    };
    let layout = match descriptor.physical_layout.as_str() {
        "row_major" => 1,
        "output_rows" => 2,
        crate::GGML_OUTPUT_ROWS_LAYOUT => 2,
        value => return Err(invalid(format!("unsupported tensor layout {value}"))),
    };
    Ok((dtype, dtype, scheme, layout))
}

fn repack_q4_0(bytes: &[u8]) -> Vec<u8> {
    const PACKED_BYTES: usize = Q4_0_BLOCK_SIZE / 2;
    const BLOCK_BYTES: usize = 2 + PACKED_BYTES;
    let mut result = bytes.to_vec();
    for (source, target) in bytes
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(result.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        let pairs = &source[2..];
        for (index, byte) in target[2..].iter_mut().enumerate() {
            let shift = (index % 2) * 4;
            let low = (pairs[index / 2] >> shift) & 0x0f;
            let high = (pairs[index / 2 + PACKED_BYTES / 2] >> shift) & 0x0f;
            *byte = low | (high << 4);
        }
    }
    result
}
