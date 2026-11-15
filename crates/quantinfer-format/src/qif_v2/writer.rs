use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use sha2::{Digest, Sha256};

use quantinfer_core::{
    DType, QuantizationScheme, Shape, TensorStorageDescriptor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
};

use crate::gguf::GgufReader;
use crate::{GgmlType, GgufValue, ModelManifest, Result, TensorManifest};

use super::metadata::QifV2MetadataValue;
use super::{
    align, invalid, metadata_from_gguf, ALIAS_NONE, ALIGNMENT, CODEC_NONE, HASH_SHA256, HASH_SIZE,
    HEADER_SIZE, INDEX_MAGIC, INDEX_VERSION, MAGIC, MAX_CHUNKS, MAX_INDEX_BYTES, MAX_RANK,
    MAX_TENSORS, VERSION,
};

/// Transcode a supported GGUF model to the new QIF v2 wire format without
/// dequantizing or requantizing its tensor bytes.
pub fn import_gguf(
    reader: &GgufReader,
    model_id: impl Into<String>,
    output: &mut impl Write,
) -> Result<()> {
    let mut metadata = reader.metadata().clone();
    let architecture = match metadata.get("general.architecture") {
        Some(GgufValue::String(architecture)) => architecture.as_str(),
        None => return Err(invalid("source GGUF is missing general.architecture")),
        Some(_) => return Err(invalid("source architecture must be a string")),
    };
    let mut manifest = ModelManifest::new(model_id, architecture);
    let ggml = reader.uses_ggml_output_rows()?;
    metadata.insert(
        "quantinfer.qif.ggml_dimension_order".to_owned(),
        GgufValue::Bool(ggml),
    );
    let mut payloads = BTreeMap::<String, Cow<'_, [u8]>>::new();
    let mut schemes = Vec::new();
    for tensor in reader.tensors() {
        let bytes = reader.tensor_bytes(&tensor.name)?;
        let (dtype, scheme, block_size) = descriptor_parts(tensor.dtype)?;
        let mut shape = tensor.shape.clone();
        if ggml {
            shape.reverse();
        }
        let shape = Shape::new(shape)?;
        let descriptor = if scheme == QuantizationScheme::None {
            TensorStorageDescriptor::floating(dtype, shape)?
        } else {
            if !schemes.contains(&scheme) {
                schemes.push(scheme);
            }
            TensorStorageDescriptor::quantized(dtype, scheme, shape, block_size)?
        };
        let payload = if dtype == DType::Q4_0 && reader.uses_legacy_quant_block_layout()? {
            Cow::Owned(repack_q4_0(bytes))
        } else {
            Cow::Borrowed(bytes)
        };
        payloads.insert(tensor.name.clone(), payload);
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
    write_qif_v2(
        manifest,
        &metadata_from_gguf(&metadata),
        |name| Ok(payloads[name].as_ref()),
        output,
    )
}

/// Write a QIF v2 single-file artifact. v2.0 emits uncompressed chunks; the
/// reader and index are already chunk-aware so compression can be added under
/// a negotiated codec in a later compatible profile.
pub fn write_qif_v2<'a>(
    mut manifest: ModelManifest,
    metadata: &BTreeMap<String, QifV2MetadataValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
) -> Result<()> {
    if manifest.tensors.len() > MAX_TENSORS {
        return Err(invalid("tensor count exceeds the limit"));
    }
    manifest.validate()?;
    let metadata_bytes = super::metadata::encode(metadata)?;

    let mut payloads = Vec::with_capacity(manifest.tensors.len());
    for tensor in &mut manifest.tensors {
        let payload = tensor_bytes(&tensor.name)?;
        let expected = payload_size(&tensor.descriptor)?;
        if payload.len() != expected {
            return Err(invalid(format!(
                "tensor {} has {} bytes, expected {expected}",
                tensor.name,
                payload.len()
            )));
        }
        validate_payload(tensor.descriptor.dtype, payload, &tensor.name)?;
        tensor.checksum_sha256 = Some(crate::sha256_hex(payload));
        payloads.push(payload);
    }
    // The manifest contains tensor checksums, so serialize it only after all
    // payload validation has completed.
    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| invalid(format!("cannot serialize manifest: {error}")))?;
    if manifest_bytes.len() > 64 * 1024 * 1024 {
        return Err(invalid("manifest exceeds the size limit"));
    }
    let (index, chunk_offsets) = build_index(&manifest, &payloads)?;
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
    let payload_offset = align(
        index_offset
            .checked_add(index.len())
            .ok_or_else(|| invalid("index range overflows"))?,
    )?;
    let mut payload_length = 0usize;
    for payload in &payloads {
        payload_length = payload_length
            .checked_add(payload.len())
            .ok_or_else(|| invalid("payload length overflows"))?;
        payload_length = align(payload_length)?;
    }

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
    write_padding(output, index_offset + index.len(), payload_offset)?;
    let mut current = payload_offset;
    for (payload, offset) in payloads.iter().zip(chunk_offsets) {
        let expected = payload_offset
            .checked_add(offset)
            .ok_or_else(|| invalid("payload offset overflows"))?;
        write_padding(output, current, expected)?;
        output.write_all(payload).map_err(write_error)?;
        current = expected
            .checked_add(payload.len())
            .ok_or_else(|| invalid("payload end overflows"))?;
    }
    let payload_end = payload_offset
        .checked_add(payload_length)
        .ok_or_else(|| invalid("payload range overflows"))?;
    write_padding(output, current, payload_end)?;
    Ok(())
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
    header.extend_from_slice(&0_u64.to_le_bytes());
    header.extend_from_slice(&0_u64.to_le_bytes());
    header.extend_from_slice(&0_u64.to_le_bytes());
    if header.len() != HEADER_SIZE {
        return Err(invalid("internal header size mismatch"));
    }
    Ok(header)
}

fn build_index(manifest: &ModelManifest, payloads: &[&[u8]]) -> Result<(Vec<u8>, Vec<usize>)> {
    let mut strings = Vec::new();
    let mut shapes = Vec::new();
    let mut records = Vec::new();
    let mut chunks = Vec::new();
    let mut names = BTreeSet::new();
    let mut offsets = Vec::with_capacity(payloads.len());
    let mut payload_offset = 0usize;
    for (index, tensor) in manifest.tensors.iter().enumerate() {
        if !names.insert(tensor.name.as_str()) {
            return Err(invalid("tensor names must be unique"));
        }
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
        let chunk_start = chunks.len() / 72;
        let payload = payloads[index];
        let hash: [u8; HASH_SIZE] = Sha256::digest(payload).into();
        let chunk = encode_chunk(
            0,
            CODEC_NONE,
            payload_offset,
            payload.len(),
            payload_offset,
            payload.len(),
            hash,
        )?;
        chunks.extend_from_slice(&chunk);
        let record = encode_tensor_record(
            index as u64,
            u32::try_from(name_offset).map_err(|_| invalid("name table offset overflows u32"))?,
            u32::try_from(shape_offset).map_err(|_| invalid("shape table offset overflows u32"))?,
            dtype,
            encoding,
            scheme,
            layout,
            CODEC_NONE,
            u32::try_from(chunk_start).map_err(|_| invalid("chunk index overflows u32"))?,
            1,
            payload.len(),
            payload.len(),
            payload_offset,
            hash,
        )?;
        records.extend_from_slice(&record);
        offsets.push(payload_offset);
        payload_offset = align(
            payload_offset
                .checked_add(payload.len())
                .ok_or_else(|| invalid("payload offset overflows"))?,
        )?;
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
    Ok((index, offsets))
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
    write_u64(&mut record, &mut offset, ALIAS_NONE);
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
    crate::FormatError::Io(format!("cannot write QIF v2: {error}"))
}

fn payload_size(descriptor: &TensorStorageDescriptor) -> Result<usize> {
    descriptor.validate()?;
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
    descriptor.validate()?;
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
        value => return Err(invalid(format!("unsupported tensor layout {value}"))),
    };
    Ok((dtype, dtype, scheme, layout))
}

fn validate_payload(dtype: DType, bytes: &[u8], name: &str) -> Result<()> {
    use half::{bf16, f16};
    let finite = |value: f32| {
        if value.is_finite() {
            Ok(())
        } else {
            Err(invalid(format!("tensor {name} contains non-finite values")))
        }
    };
    match dtype {
        DType::F32 => {
            for chunk in bytes.as_chunks::<4>().0 {
                finite(f32::from_le_bytes(*chunk))?;
            }
        }
        DType::F16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                finite(f16::from_le_bytes(*chunk).to_f32())?;
            }
        }
        DType::BF16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                finite(bf16::from_le_bytes(*chunk).to_f32())?;
            }
        }
        DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
            let (block_bytes, scale_offsets): (usize, &[usize]) = match dtype {
                DType::Q8_0 => (34, &[0]),
                DType::Q4_0 => (18, &[0]),
                DType::Q4_K => (144, &[0, 2]),
                DType::Q5_K => (176, &[0, 2]),
                DType::Q6_K => (210, &[208]),
                DType::F32 | DType::F16 | DType::BF16 => unreachable!(),
            };
            for block in bytes.chunks_exact(block_bytes) {
                for &offset in scale_offsets {
                    let scale =
                        f16::from_le_bytes(block[offset..offset + 2].try_into().unwrap()).to_f32();
                    finite(scale)?;
                }
            }
        }
    }
    Ok(())
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
