//! Shared command-line parsing and output helpers.

pub(crate) fn parse_token_ids(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|token| {
            token
                .trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid token id {token:?}: {error}"))
        })
        .collect()
}

pub(crate) fn last_logits(
    logits: &quantinfer_core::Tensor,
) -> Result<quantinfer_core::Tensor, String> {
    let dims = logits.shape().dims();
    if dims.len() != 2 || dims[1] == 0 || dims[0] == 0 {
        return Err("model logits must have shape [tokens, vocab]".to_owned());
    }
    let values = logits
        .as_f32()
        .map_err(|error| format!("logits must be F32: {error}"))?;
    let vocab_size = dims[1];
    let start = (dims[0] - 1) * vocab_size;
    quantinfer_core::Tensor::from_f32(vec![vocab_size], values[start..].to_vec())
        .map_err(|error| format!("cannot build last-logits tensor: {error}"))
}

pub(crate) fn option_value(arguments: &[String], flag: &str) -> Result<Option<String>, String> {
    let Some(index) = arguments.iter().position(|argument| argument == flag) else {
        return Ok(None);
    };
    arguments
        .get(index + 1)
        .cloned()
        .map(Some)
        .ok_or_else(|| format!("{flag} requires a value"))
}

pub(crate) fn has_flag(arguments: &[String], flag: &str) -> bool {
    arguments.iter().any(|argument| argument == flag)
}

pub(crate) fn print_help() {
    println!(
        "quantinfer commands:\n  render-chat --text TEXT [--role ROLE] [--think]\n  inspect-gguf --input MODEL.gguf|MODEL_DIR [--json]\n  verify-package --input MODEL_DIR [--json]\n  list-profiles --input MODEL_DIR [--json]\n  collect-imatrix --input MODEL.gguf|MODEL.qif|MODEL_DIR --dataset PROMPTS.jsonl --output IMATRIX.json [--max-sequence-length N]\n  generate --input MODEL.gguf|MODEL.qif|MODEL_DIR (--tokens 1,2,3 | --prompt TEXT) [--backend cpu|wgpu|cuda|metal] [--lora ADAPTER_DIR] [--chat] [--role ROLE] [--think] [--max-new-tokens N] [--max-sequence-length N] [--memory-budget-bytes N] [--inference-profile NAME] [--temperature T] [--seed N] [--eos-token-id ID] [--profile]"
    );
}
