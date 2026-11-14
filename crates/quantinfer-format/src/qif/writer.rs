use super::*;

/// Write a QIF artifact to a stream. The manifest provides tensor order and
/// descriptors, and the writer computes the SHA-256 for each payload.
pub fn write_qif<'a>(
    mut manifest: ModelManifest,
    metadata: &BTreeMap<String, GgufValue>,
    mut tensor_bytes: impl FnMut(&str) -> Result<&'a [u8]>,
    output: &mut impl Write,
) -> Result<()> {
    manifest.validate()?;
    let mut payloads = Vec::with_capacity(manifest.tensors.len());
    for tensor in &mut manifest.tensors {
        let payload = tensor_bytes(&tensor.name)?;
        if payload.len() != payload_size(&tensor.descriptor)? {
            return Err(invalid(format!(
                "tensor {} has an invalid length",
                tensor.name
            )));
        }
        tensor.checksum_sha256 = Some(sha256_hex(payload));
        payloads.push(payload);
    }
    let manifest = serde_json::to_vec(&manifest)
        .map_err(|error| invalid(format!("cannot serialize manifest: {error}")))?;
    let metadata = write_metadata(metadata)?;
    let header_size = manifest.len().saturating_add(metadata.len());
    if header_size > MAX_HEADER_BYTES {
        return Err(invalid("header exceeds the size limit"));
    }
    let mut digest = Sha256::new();
    digest.update(&manifest);
    digest.update(&metadata);
    let checksum = digest.finalize();

    let mut end = PREFIX + header_size;
    output
        .write_all(MAGIC)
        .and_then(|()| output.write_all(&VERSION.to_le_bytes()))
        .and_then(|()| output.write_all(&(manifest.len() as u64).to_le_bytes()))
        .and_then(|()| output.write_all(&(metadata.len() as u64).to_le_bytes()))
        .and_then(|()| output.write_all(&checksum))
        .and_then(|()| output.write_all(&manifest))
        .and_then(|()| output.write_all(&metadata))
        .map_err(write_error)?;
    for payload in payloads {
        let start = aligned(end)?;
        let padding = start - end;
        end = start
            .checked_add(payload.len())
            .ok_or_else(|| invalid("tensor end overflows usize"))?;
        output
            .write_all(&[0; ALIGNMENT][..padding])
            .and_then(|()| output.write_all(payload))
            .map_err(write_error)?;
    }
    Ok(())
}
