use std::env;
use std::fs;
use std::path::PathBuf;

use quantinfer_core::Tensor;
use quantinfer_format::{GgufValue, GgufWriter, SafeTensorFile};
use quantinfer_model::Qwen3Config;
use quantinfer_quant::quantize_q8_0;

fn main() {
    if let Err(error) = run() {
        eprintln!("quantinfer-convert: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print_help();
        return Ok(());
    }
    let config_path = required_path(&arguments, "--config")?;
    let weights_path = required_path(&arguments, "--weights")?;
    let output_path = required_path(&arguments, "--output")?;
    let quantization = optional_value(&arguments, "--quantize");
    if let Some(value) = &quantization {
        if value != "q8_0" {
            return Err(format!("unsupported quantization {value}; use q8_0"));
        }
    }

    let config_json = fs::read_to_string(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    let config = Qwen3Config::from_json(&config_json)
        .map_err(|error| format!("cannot parse Qwen3 config: {error}"))?;
    config
        .validate()
        .map_err(|error| format!("invalid Qwen3 config: {error}"))?;
    let weights = SafeTensorFile::open(
        fs::read(&weights_path)
            .map_err(|error| format!("cannot read {}: {error}", weights_path.display()))?,
    )
    .map_err(|error| format!("cannot parse SafeTensors: {error}"))?;

    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("qwen3".to_owned()),
    );
    writer.insert_metadata(
        "qwen3.context_length",
        GgufValue::U32(metadata_u32(config.max_position_embeddings, "max_position_embeddings")?),
    );
    writer.insert_metadata(
        "qwen3.embedding_length",
        GgufValue::U32(metadata_u32(config.hidden_size, "hidden_size")?),
    );
    writer.insert_metadata(
        "qwen3.vocab_size",
        GgufValue::U32(metadata_u32(config.vocab_size, "vocab_size")?),
    );
    writer.insert_metadata(
        "qwen3.feed_forward_length",
        GgufValue::U32(metadata_u32(config.intermediate_size, "intermediate_size")?),
    );
    writer.insert_metadata(
        "qwen3.block_count",
        GgufValue::U32(metadata_u32(config.num_hidden_layers, "num_hidden_layers")?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count",
        GgufValue::U32(metadata_u32(
            config.num_attention_heads,
            "num_attention_heads",
        )?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count_kv",
        GgufValue::U32(metadata_u32(
            config.num_key_value_heads,
            "num_key_value_heads",
        )?),
    );
    writer.insert_metadata(
        "qwen3.attention.head_dim",
        GgufValue::U32(metadata_u32(
            config.effective_head_dim(),
            "head_dim",
        )?),
    );
    writer.insert_metadata("qwen3.rope.freq_base", GgufValue::F32(config.rope_theta));
    writer.insert_metadata(
        "qwen3.attention.layer_norm_rms_epsilon",
        GgufValue::F32(config.rms_norm_eps),
    );
    writer.insert_metadata(
        "qwen3.tie_word_embeddings",
        GgufValue::Bool(config.tie_word_embeddings),
    );
    writer.insert_metadata(
        "qwen3.architectures",
        GgufValue::Array(
            config
                .architectures
                .iter()
                .cloned()
                .map(GgufValue::String)
                .collect(),
        ),
    );
    for (name, info) in weights.tensors() {
        let values = weights
            .tensor_f32(name)
            .map_err(|error| format!("cannot decode tensor {name}: {error}"))?;
        let canonical_name = canonical_tensor_name(name);
        let (shape, values) = canonicalize_tensor_layout(
            canonical_name.as_deref().unwrap_or(name),
            &info.shape,
            values,
        )?;
        if quantization.as_deref() == Some("q8_0")
            && is_quantizable_linear(canonical_name.as_deref().unwrap_or(name))
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q8_0_BLOCK_SIZE)
        {
            let tensor = Tensor::from_f32(shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q8_0(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q8_0_tensor(canonical_name.as_deref().unwrap_or(name), &quantized);
        } else {
            writer
                .add_f32_tensor(canonical_name.as_deref().unwrap_or(name), shape, &values)
                .map_err(|error| format!("cannot add tensor {name}: {error}"))?;
        }
    }
    let bytes = writer
        .finish()
        .map_err(|error| format!("cannot write GGUF: {error}"))?;
    fs::write(&output_path, bytes)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    println!(
        "converted {} tensors to {}",
        weights.tensors().len(),
        output_path.display()
    );
    Ok(())
}

fn required_path(arguments: &[String], flag: &str) -> Result<PathBuf, String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or_else(|| format!("missing {flag}; use --help for usage"))
}

fn optional_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn metadata_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} does not fit in GGUF u32 metadata"))
}

fn canonical_tensor_name(name: &str) -> Option<String> {
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

fn canonicalize_tensor_layout(
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if !is_matrix_weight(name) {
        return Ok((shape.to_vec(), values));
    }
    if shape.len() != 2 {
        return Err(format!("linear tensor {name} must be rank 2"));
    }
    let rows = shape[0];
    let columns = shape[1];
    if values.len() != rows * columns {
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

fn is_matrix_weight(name: &str) -> bool {
    name == "output.weight"
        || [
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

fn is_quantizable_linear(name: &str) -> bool {
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

fn print_help() {
    println!(
        "Usage: quantinfer-convert --config config.json --weights model.safetensors --output model.gguf [--quantize q8_0]"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_tensor_name, canonicalize_tensor_layout, is_matrix_weight,
        is_quantizable_linear,
    };

    #[test]
    fn maps_qwen3_huggingface_names_to_gguf_names() {
        assert_eq!(
            canonical_tensor_name("model.layers.2.self_attn.q_proj.weight"),
            Some("blk.2.attn_q.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.layers.2.mlp.down_proj.weight"),
            Some("blk.2.ffn_down.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.embed_tokens.weight"),
            Some("token_embd.weight".to_owned())
        );
        assert_eq!(
            canonical_tensor_name("model.norm.weight"),
            Some("output_norm.weight".to_owned())
        );
    }

    #[test]
    fn transposes_runtime_linear_layout() {
        let (shape, values) = canonicalize_tensor_layout(
            "blk.0.attn_q.weight",
            &[2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(is_matrix_weight("blk.0.attn_q.weight"));
        assert!(is_quantizable_linear("blk.0.attn_q.weight"));
        assert!(!is_matrix_weight("blk.0.attn_q_norm.weight"));
    }
}
