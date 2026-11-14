//! Hugging Face tokenizer metadata extraction.

use std::fs;
use std::path::Path;

use serde_json::Value;

pub(crate) struct TokenizerMetadata {
    pub(crate) tokens: Vec<String>,
    pub(crate) merges: Vec<String>,
    pub(crate) bos_token_id: Option<usize>,
    pub(crate) eos_token_id: Option<usize>,
    pub(crate) token_types: Vec<i32>,
    pub(crate) add_bos_token: bool,
    pub(crate) add_eos_token: bool,
}

pub(crate) fn load_tokenizer_metadata(
    tokenizer_path: &Path,
    config_json: &str,
) -> Result<TokenizerMetadata, String> {
    let config: Value = serde_json::from_str(config_json)
        .map_err(|error| format!("cannot parse model config for tokenizer ids: {error}"))?;
    let model_vocab_size = config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .and_then(|size| usize::try_from(size).ok())
        .filter(|size| *size > 0)
        .ok_or_else(|| "model config is missing a positive vocab_size".to_owned())?;
    let tokenizer_json = fs::read_to_string(tokenizer_path)
        .map_err(|error| format!("cannot read {}: {error}", tokenizer_path.display()))?;
    let root: Value = serde_json::from_str(&tokenizer_json)
        .map_err(|error| format!("cannot parse {}: {error}", tokenizer_path.display()))?;
    let vocab = root
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(Value::as_object)
        .ok_or_else(|| "tokenizer.json is missing model.vocab".to_owned())?;
    let maximum_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .ok_or_else(|| "tokenizer.json model.vocab is empty".to_owned())?;
    let maximum_id = usize::try_from(maximum_id)
        .map_err(|_| "tokenizer vocabulary id overflows usize".to_owned())?;
    if maximum_id >= model_vocab_size {
        return Err(format!(
            "tokenizer vocabulary id {maximum_id} exceeds model vocabulary {model_vocab_size}"
        ));
    }
    let token_count = maximum_id
        .checked_add(1)
        .ok_or_else(|| "tokenizer vocabulary size overflows usize".to_owned())?;
    let mut tokens = vec![None; token_count];
    let mut token_types = vec![1; token_count];
    for (token, value) in vocab {
        let token_id = value
            .as_u64()
            .ok_or_else(|| format!("tokenizer id for {token:?} is not an integer"))?;
        let token_id = usize::try_from(token_id)
            .map_err(|_| format!("tokenizer id for {token:?} overflows usize"))?;
        if token_id >= tokens.len() || tokens[token_id].is_some() {
            return Err(format!(
                "tokenizer vocabulary contains duplicate id {token_id}"
            ));
        }
        tokens[token_id] = Some(token.clone());
    }
    if let Some(added_tokens) = root.get("added_tokens").and_then(Value::as_array) {
        for added_token in added_tokens {
            let Some(token) = added_token.get("content").and_then(Value::as_str) else {
                continue;
            };
            let Some(token_id) = added_token.get("id").and_then(Value::as_u64) else {
                continue;
            };
            let token_id = usize::try_from(token_id)
                .map_err(|_| format!("added tokenizer id for {token:?} overflows usize"))?;
            if token_id >= model_vocab_size {
                return Err(format!(
                    "added tokenizer id {token_id} exceeds model vocabulary {model_vocab_size}"
                ));
            }
            if token_id >= tokens.len() {
                tokens.resize(token_id + 1, None);
                token_types.resize(token_id + 1, 1);
            }
            if tokens[token_id].is_some() {
                return Err(format!(
                    "tokenizer vocabulary contains duplicate id {token_id}"
                ));
            }
            tokens[token_id] = Some(token.to_owned());
            token_types[token_id] =
                if added_token.get("special").and_then(Value::as_bool) == Some(true) {
                    3 // GGML control token
                } else {
                    4 // GGML user-defined token
                };
        }
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(token_id, token)| {
            token.ok_or_else(|| format!("tokenizer vocabulary is missing id {token_id}"))
        })
        .collect::<Result<_, _>>()?;
    let merges = root
        .get("model")
        .and_then(|model| model.get("merges"))
        .and_then(Value::as_array)
        .map(|merges| {
            merges
                .iter()
                .map(|merge| {
                    if let Some(merge) = merge.as_str() {
                        Ok(merge.to_owned())
                    } else if let Some(parts) = merge.as_array() {
                        if parts.len() != 2 {
                            return Err(
                                "tokenizer merge arrays must contain two strings".to_owned()
                            );
                        }
                        let left = parts[0].as_str().ok_or_else(|| {
                            "tokenizer merge left side is not a string".to_owned()
                        })?;
                        let right = parts[1].as_str().ok_or_else(|| {
                            "tokenizer merge right side is not a string".to_owned()
                        })?;
                        Ok(format!("{left} {right}"))
                    } else {
                        Err("tokenizer merge must be a string or two-item array".to_owned())
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let bos_token_id = token_id_from_value(config.get("bos_token_id"));
    let eos_token_id = token_id_from_value(config.get("eos_token_id"));
    let template = root["post_processor"]["single"].as_array();
    let matches_special = |entry: Option<&Value>, token: Option<usize>| {
        let Some(text) = token.and_then(|id| tokens.get(id)) else {
            return false;
        };
        entry.and_then(|entry| entry["SpecialToken"]["id"].as_str()) == Some(text.as_str())
    };
    let add_bos_token = matches_special(template.and_then(|items| items.first()), bos_token_id);
    let add_eos_token = matches_special(template.and_then(|items| items.last()), eos_token_id);
    Ok(TokenizerMetadata {
        tokens,
        merges,
        bos_token_id,
        eos_token_id,
        token_types,
        add_bos_token,
        add_eos_token,
    })
}

fn token_id_from_value(value: Option<&Value>) -> Option<usize> {
    value.and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_array()?.first()?.as_u64())
            .and_then(|id| usize::try_from(id).ok())
    })
}
