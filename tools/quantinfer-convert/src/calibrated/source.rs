//! Source controls and snapshot identity; no intermediate weight container.

use quantinfer_format::{qif, sha256_hex, GgmlType, GgufReader, ModelManifest};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(super) type Metadata = BTreeMap<String, qif::MetadataValue>;

pub(super) fn describe(
    reader: &GgufReader,
    external: Option<&ModelManifest>,
) -> Result<(ModelManifest, Metadata, Value), String> {
    for tensor in reader.tensors() {
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(format!(
                "GPTQ requires a floating source; {} is already quantized",
                tensor.name
            ));
        }
    }
    let (mut manifest, metadata) = if let Some(manifest) = reader.embedded_manifest() {
        reader.verify_qif().map_err(|e| e.to_string())?;
        (
            manifest.clone(),
            reader
                .qif_metadata()
                .ok_or("missing native QIF metadata")?
                .clone(),
        )
    } else {
        qif::describe_gguf(reader, "source").map_err(|e| e.to_string())?
    };
    if let Some(original) = external {
        reader
            .validate_manifest(original)
            .map_err(|e| e.to_string())?;
        manifest.source_revision = original.source_revision.clone();
        manifest.tokenizer_revision = original.tokenizer_revision.clone();
        manifest.compatibility = original.compatibility.clone();
        manifest.kernel_requirements = original.kernel_requirements.clone();
    }
    let mut tensors = BTreeMap::new();
    for info in reader.tensors() {
        if reader.qif_metadata().is_none() {
            reader.tensor_f32(&info.name).map_err(|e| e.to_string())?;
        }
        let raw = reader.tensor_data(&info.name).map_err(|e| e.to_string())?;
        tensors.insert(info.name.clone(),json!({"compatibility_shape":info.shape,"dtype":format!("{:?}",info.dtype),"sha256":sha256_hex(raw.as_ref())}));
    }
    let encoded = serde_json::to_vec(&(&manifest, &metadata, &tensors, external))
        .map_err(|e| e.to_string())?;
    let identity = json!({"inventory_sha256":sha256_hex(&encoded),"tensors":tensors,"original_manifest":manifest,"external_manifest":external});
    manifest.checksums.clear();
    if reader.qif_metadata().is_some() {
        manifest.source_format = "qif".into();
    }
    Ok((manifest, metadata, identity))
}
