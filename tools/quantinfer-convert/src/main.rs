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
        GgufValue::U32(config.max_position_embeddings as u32),
    );
    writer.insert_metadata(
        "qwen3.embedding_length",
        GgufValue::U32(config.hidden_size as u32),
    );
    writer.insert_metadata(
        "qwen3.block_count",
        GgufValue::U32(config.num_hidden_layers as u32),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count",
        GgufValue::U32(config.num_attention_heads as u32),
    );
    writer.insert_metadata(
        "qwen3.attention.head_count_kv",
        GgufValue::U32(config.num_key_value_heads as u32),
    );
    writer.insert_metadata(
        "qwen3.attention.head_dim",
        GgufValue::U32(config.effective_head_dim() as u32),
    );
    writer.insert_metadata("qwen3.rope.freq_base", GgufValue::F32(config.rope_theta));
    for (name, info) in weights.tensors() {
        let values = weights
            .tensor_f32(name)
            .map_err(|error| format!("cannot decode tensor {name}: {error}"))?;
        if quantization.as_deref() == Some("q8_0")
            && is_quantizable_linear(name)
            && values
                .len()
                .is_multiple_of(quantinfer_core::Q8_0_BLOCK_SIZE)
        {
            let tensor = Tensor::from_f32(info.shape.clone(), values)
                .map_err(|error| format!("cannot build tensor {name}: {error}"))?;
            let quantized = quantize_q8_0(&tensor)
                .map_err(|error| format!("cannot quantize tensor {name}: {error}"))?;
            writer.add_q8_0_tensor(name, &quantized);
        } else {
            writer
                .add_f32_tensor(name, info.shape.clone(), &values)
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

fn is_quantizable_linear(name: &str) -> bool {
    [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(suffix))
}

fn print_help() {
    println!(
        "Usage: quantinfer-convert --config config.json --weights model.safetensors --output model.gguf [--quantize q8_0]"
    );
}
