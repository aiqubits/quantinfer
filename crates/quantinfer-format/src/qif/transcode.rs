use super::*;

/// Transcode a supported GGUF model without dequantizing or requantizing.
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
    metadata.insert(GGML_DIMENSION_ORDER.to_owned(), GgufValue::Bool(ggml));
    let mut payloads = BTreeMap::new();
    let mut schemes = Vec::new();
    for tensor in reader.tensors() {
        let bytes = reader.tensor_bytes(&tensor.name)?;
        let (dtype, scheme, block_size) = match tensor.dtype {
            GgmlType::F32 => (DType::F32, QuantizationScheme::None, 0),
            GgmlType::F16 => (DType::F16, QuantizationScheme::None, 0),
            GgmlType::BF16 => (DType::BF16, QuantizationScheme::None, 0),
            GgmlType::Q8_0 => (DType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
            GgmlType::Q4_0 => (DType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
            GgmlType::Q4_K => (DType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q5_K => (DType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q6_K => (DType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
        };
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
        payloads.insert(tensor.name.as_str(), payload);
        manifest.tensors.push(TensorManifest {
            name: tensor.name.clone(),
            descriptor,
            checksum_sha256: None,
        });
    }
    manifest.quantization = if matches!(
        metadata.get("quantinfer.quantization.variant"),
        Some(GgufValue::String(variant)) if variant == "mixed"
    ) {
        QuantizationScheme::Mixed
    } else {
        match schemes.as_slice() {
            [] => QuantizationScheme::None,
            [scheme] => *scheme,
            _ => QuantizationScheme::Mixed,
        }
    };
    write_qif(
        manifest,
        &metadata,
        |name| Ok(payloads[name].as_ref()),
        output,
    )
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
