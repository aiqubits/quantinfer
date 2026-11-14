//! Canonical dense tensor names and checkpoint layout transforms.
//!
//! These helpers describe the GGUF dense storage boundary without depending on
//! a model architecture crate. Architecture crates expose semantic roles;
//! callers at the format boundary decide how source checkpoint names and
//! matrix layouts are encoded.

/// Map common Hugging Face dense checkpoint names to canonical GGUF names.
pub fn canonical_dense_tensor_name(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('.').collect();
    match parts.as_slice() {
        ["model", "embed_tokens", "weight"] | ["model", "token_embeddings", "weight"] => {
            Some("token_embd.weight".to_owned())
        }
        ["model", "norm", "weight"] => Some("output_norm.weight".to_owned()),
        ["lm_head", "weight"] => Some("output.weight".to_owned()),
        ["model", "layers", index, "input_layernorm", "weight"] => {
            Some(format!("blk.{index}.attn_norm.weight"))
        }
        ["model", "layers", index, "self_attn", projection, "weight"] => {
            let suffix = match *projection {
                "q_proj" => "attn_q",
                "q_norm" => "attn_q_norm",
                "k_proj" => "attn_k",
                "k_norm" => "attn_k_norm",
                "v_proj" => "attn_v",
                "o_proj" => "attn_output",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        ["model", "layers", index, "post_attention_layernorm", "weight"] => {
            Some(format!("blk.{index}.ffn_norm.weight"))
        }
        ["model", "layers", index, "mlp", projection, "weight"] => {
            let suffix = match *projection {
                "gate_proj" => "ffn_gate",
                "up_proj" => "ffn_up",
                "down_proj" => "ffn_down",
                _ => return None,
            };
            Some(format!("blk.{index}.{suffix}.weight"))
        }
        _ => None,
    }
}

/// Return whether a canonical dense tensor stores a linear matrix.
pub fn is_matrix_storage_name(name: &str) -> bool {
    is_lm_head_storage_name(name) || is_linear_projection_storage_name(name)
}

/// Return whether a canonical dense tensor is an embedding table.
pub fn is_embedding_storage_name(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Return whether a canonical dense tensor is the language-model head.
pub fn is_lm_head_storage_name(name: &str) -> bool {
    name == "output.weight"
}

/// Return whether a canonical dense tensor is a quantizable projection.
pub fn is_linear_projection_storage_name(name: &str) -> bool {
    [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

/// Convert a source matrix from row-major output/input order to GGUF's
/// canonical input/output order.
pub fn transpose_dense_layout(
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if !is_matrix_storage_name(name) {
        return Ok((shape.to_vec(), values));
    }
    if shape.len() != 2 {
        return Err(format!("linear tensor {name} must be rank 2"));
    }
    let rows = shape[0];
    let columns = shape[1];
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| format!("linear tensor {name} shape overflows usize"))?;
    if values.len() != elements {
        return Err(format!("linear tensor {name} shape does not match data"));
    }
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }
    Ok((vec![columns, rows], transposed))
}

#[cfg(test)]
mod tests {
    use super::{canonical_dense_tensor_name, transpose_dense_layout};

    #[test]
    fn maps_dense_checkpoint_names_and_transposes_linear_weights() {
        assert_eq!(
            canonical_dense_tensor_name("model.layers.3.self_attn.q_proj.weight"),
            Some("blk.3.attn_q.weight".to_owned())
        );
        let (shape, values) = transpose_dense_layout(
            "blk.3.attn_q.weight",
            &[2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(shape, [3, 2]);
        assert_eq!(values, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn leaves_non_matrix_storage_in_source_layout() {
        let result =
            transpose_dense_layout("blk.0.attn_norm.weight", &[3], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result, (vec![3], vec![1.0, 2.0, 3.0]));
    }
}
