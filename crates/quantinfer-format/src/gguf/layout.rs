//! GGML physical layout helpers.

use super::{FormatError, PackedTensorReader, Result, GGML_OUTPUT_ROWS_LAYOUT};

impl PackedTensorReader {
    /// The recorded storage layout, independent of the GGUF-compatible shape
    /// projection. QIF keeps its per-tensor descriptor; GGUF derives its uniform
    /// matrix convention from metadata. This method reads no tensor payload.
    pub fn tensor_storage_layout(&self, name: &str) -> Result<&'static str> {
        let info = self.tensor_info(name)?;
        let layout = if let Some(manifest) = self.embedded_manifest() {
            manifest
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .map(|tensor| tensor.descriptor.physical_layout.as_str())
                .ok_or_else(|| FormatError::Invalid(format!("tensor {name} has no descriptor")))?
        } else if info.shape.len() == 2
            && self.uses_ggml_output_rows()?
            && !crate::is_embedding_storage_name(name)
        {
            GGML_OUTPUT_ROWS_LAYOUT
        } else {
            "row_major"
        };
        match layout {
            "row_major" => Ok("row_major"),
            GGML_OUTPUT_ROWS_LAYOUT | "output_rows" => Ok(GGML_OUTPUT_ROWS_LAYOUT),
            _ => Err(FormatError::Invalid(format!(
                "unsupported tensor layout {layout:?} for {name} with rank {}",
                info.shape.len()
            ))),
        }
    }

    /// Whether matrix bytes form output rows in `tensor_info().shape`, the
    /// compatibility view used by dense kernels. Native QIF uses its tensor
    /// descriptor. In the GGML dimension-order profile the view reverses the
    /// logical axes, so row-major logical [N,K] also forms output rows in [K,N].
    pub fn matrix_uses_output_rows(&self, name: &str) -> Result<bool> {
        if self.tensor_info(name)?.shape.len() != 2 {
            return Err(FormatError::Invalid(format!(
                "tensor {name} must be a matrix"
            )));
        }
        let layout = self.tensor_storage_layout(name)?;
        Ok(self.uses_ggml_output_rows()? || layout == GGML_OUTPUT_ROWS_LAYOUT)
    }
}

/// Map an adjacent-pair GGML Q/K row to the checkpoint's half-split channel.
pub fn qk_channel(row: usize, head_dim: usize) -> usize {
    let channel = row % head_dim;
    row - channel + channel / 2 + (channel % 2) * (head_dim / 2)
}

#[cfg(test)]
mod tests {
    use super::qk_channel;

    #[test]
    fn maps_adjacent_pairs_to_half_split_channels() {
        assert_eq!(
            (0..8).map(|row| qk_channel(row, 8)).collect::<Vec<_>>(),
            [0, 4, 1, 5, 2, 6, 3, 7]
        );
        assert_eq!(qk_channel(9, 8), 12);
    }

    #[test]
    fn qif_layouts_are_structural_and_external_manifests_cannot_override_them() {
        use super::*;
        use crate::{qif, ModelManifest, TensorManifest};
        use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
        use std::collections::BTreeMap;
        for ggml in [false, true] {
            for rows in [false, true] {
                let mut descriptor =
                    TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2, 3])).unwrap();
                descriptor.physical_layout = if rows {
                    GGML_OUTPUT_ROWS_LAYOUT
                } else {
                    "row_major"
                }
                .into();
                let mut manifest = ModelManifest::new("layout-info", "llama");
                manifest.tensors.push(TensorManifest {
                    name: "weight".into(),
                    descriptor,
                    checksum_sha256: None,
                });
                let metadata = BTreeMap::from([(
                    "quantinfer.qif.ggml_dimension_order".into(),
                    qif::MetadataValue::Bool(ggml),
                )]);
                let mut bytes = Vec::new();
                qif::write_qif(manifest, &metadata, |_| Ok(&[0; 24]), &mut bytes).unwrap();
                let reader = PackedTensorReader::open_artifact(bytes).unwrap();
                assert_eq!(
                    reader.matrix_uses_output_rows("weight").unwrap(),
                    ggml || rows
                );
                assert_eq!(
                    reader.tensor_storage_layout("weight").unwrap(),
                    if rows {
                        GGML_OUTPUT_ROWS_LAYOUT
                    } else {
                        "row_major"
                    }
                );
                reader
                    .validate_manifest(reader.embedded_manifest().unwrap())
                    .unwrap();
                let mut wrong = reader.embedded_manifest().unwrap().clone();
                wrong.tensors[0].descriptor.physical_layout = if rows {
                    "row_major"
                } else {
                    GGML_OUTPUT_ROWS_LAYOUT
                }
                .into();
                assert!(reader
                    .validate_manifest(&wrong)
                    .unwrap_err()
                    .to_string()
                    .contains("layout"));
            }
        }
    }
}
