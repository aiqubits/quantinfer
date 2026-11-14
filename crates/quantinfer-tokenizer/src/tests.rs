use super::{
    ByteTokenizer, ChatMessage, ChatTemplate, ChatTemplateOptions, GgufBpeTokenizer,
    Qwen3ChatTemplate, ThinkingMode, Tokenizer,
};

#[test]
fn byte_tokenizer_round_trips_utf8_text() {
    let tokenizer = ByteTokenizer;
    let ids = tokenizer.encode("QuantInfer").unwrap();
    assert_eq!(tokenizer.decode(&ids).unwrap(), "QuantInfer");
    assert_eq!(tokenizer.vocab_size(), 259);
}

#[test]
fn renders_qwen3_non_thinking_generation_prompt() {
    let prompt = Qwen3ChatTemplate
        .render(
            &[ChatMessage::new("user", "Hello")],
            ThinkingMode::Disabled,
            true,
        )
        .unwrap();
    assert!(prompt.contains("<|im_start|>user"));
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn renders_multi_turn_history_and_tool_messages() {
    let prompt = Qwen3ChatTemplate
        .render_messages(
            &[
                ChatMessage::new("system", "You are concise."),
                ChatMessage::new("user", "Call the tool."),
                ChatMessage::new("assistant", "<tool_call>{}</tool_call>"),
                ChatMessage::new("tool", "{}").clone(),
            ],
            ChatTemplateOptions {
                thinking: ThinkingMode::Enabled,
                add_generation_prompt: true,
                add_bos_token: true,
            },
        )
        .unwrap();
    assert!(prompt.starts_with("<|im_start|><|im_start|>system\n"));
    assert!(prompt.contains("<|im_start|>tool\n{}<|im_end|>\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn gguf_bpe_tokenizer_round_trips_byte_level_text() {
    let tokenizer = GgufBpeTokenizer::new(
        vec![
            "h".to_owned(),
            "i".to_owned(),
            "hi".to_owned(),
            "Ġ".to_owned(),
            "t".to_owned(),
            "h".to_owned(),
            "e".to_owned(),
            "th".to_owned(),
            "Ġth".to_owned(),
            "Ġthe".to_owned(),
            "<|im_end|>".to_owned(),
        ],
        &[
            "h i".to_owned(),
            "t h".to_owned(),
            "Ġ th".to_owned(),
            "Ġth e".to_owned(),
        ],
        None,
        Some(10),
    )
    .unwrap();

    let ids = tokenizer.encode("hi the<|im_end|>").unwrap();
    assert_eq!(ids, vec![2, 9, 10]);
    assert_eq!(tokenizer.decode(&ids).unwrap(), "hi the");
}

#[test]
fn validates_gguf_pre_tokenizer_modes() {
    let tokens = vec!["a".to_owned()];
    assert!(GgufBpeTokenizer::new_with_pre(tokens.clone(), &[], None, None, Some("qwen2")).is_ok());
    assert!(GgufBpeTokenizer::new_with_pre(tokens, &[], None, None, Some("unsupported")).is_err());
}

#[test]
fn byte_level_pretokenization_prevents_cross_word_merges() {
    let tokenizer = GgufBpeTokenizer::new(
        vec![
            "a".to_owned(),
            "Ġ".to_owned(),
            "b".to_owned(),
            "aĠ".to_owned(),
            "Ġb".to_owned(),
        ],
        &["a Ġ".to_owned(), "Ġ b".to_owned()],
        None,
        None,
    )
    .unwrap();

    assert_eq!(tokenizer.encode("a b").unwrap(), vec![0, 4]);
}
