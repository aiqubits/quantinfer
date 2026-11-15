//! Official tokenizer golden-vector validation task.

use std::fs;

use quantinfer_tokenizer::{GgufBpeTokenizer, Tokenizer};
use serde_json::Value;

use crate::option_value;

pub(crate) fn run() {
    let tokenizer_path =
        option_value("--tokenizer").expect("tokenizer-golden requires --tokenizer");
    let golden_path = option_value("--golden").expect("tokenizer-golden requires --golden");
    let root: Value =
        serde_json::from_str(&fs::read_to_string(tokenizer_path).expect("read tokenizer.json"))
            .expect("parse tokenizer.json");
    let vocab = root["model"]["vocab"]
        .as_object()
        .expect("tokenizer model.vocab");
    let maximum_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .expect("non-empty vocabulary") as usize;
    let mut tokens = vec![String::new(); maximum_id + 1];
    for (token, id) in vocab {
        tokens[id.as_u64().expect("integer token id") as usize] = token.clone();
    }
    if let Some(added) = root["added_tokens"].as_array() {
        for entry in added {
            if let (Some(id), Some(content)) = (entry["id"].as_u64(), entry["content"].as_str()) {
                let id = id as usize;
                if id >= tokens.len() {
                    tokens.resize(id + 1, String::new());
                }
                tokens[id] = content.to_owned();
            }
        }
    }
    let merges = root["model"]["merges"]
        .as_array()
        .expect("tokenizer model.merges")
        .iter()
        .map(|merge| {
            if let Some(value) = merge.as_str() {
                value.to_owned()
            } else {
                let parts = merge.as_array().expect("merge pair");
                format!(
                    "{} {}",
                    parts[0].as_str().unwrap(),
                    parts[1].as_str().unwrap()
                )
            }
        })
        .collect::<Vec<_>>();
    let tokenizer = GgufBpeTokenizer::new(tokens, &merges, None, None).expect("build tokenizer");
    let golden: Value =
        serde_json::from_str(&fs::read_to_string(golden_path).expect("read golden vectors"))
            .expect("parse golden vectors");
    for vector in golden["vectors"].as_array().expect("golden vectors array") {
        let text = vector["text"].as_str().unwrap();
        let expected = vector["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let actual = tokenizer.encode(text).expect("encode golden text");
        assert_eq!(actual, expected, "tokenizer golden mismatch for {text:?}");
        let decoded = tokenizer.decode(&actual).expect("decode golden ids");
        assert_eq!(
            decoded,
            vector["decoded"].as_str().unwrap(),
            "tokenizer decode mismatch for {text:?}"
        );
    }
    println!(
        "tokenizer_golden_ok vectors={}",
        golden["vectors"].as_array().unwrap().len()
    );
}
