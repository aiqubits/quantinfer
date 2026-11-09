use quantinfer_tokenizer::{
    ByteTokenizer, ChatMessage, Qwen3ChatTemplate, ThinkingMode, Tokenizer,
};

#[test]
fn fixed_tokenizer_vectors_cover_text_whitespace_and_special_tokens() {
    let tokenizer = ByteTokenizer;
    let vectors = [
        (
            "QuantInfer",
            vec![81, 117, 97, 110, 116, 73, 110, 102, 101, 114],
        ),
        ("中文", vec![228, 184, 173, 230, 150, 135]),
        ("42! \n", vec![52, 50, 33, 32, 10]),
    ];
    for (text, expected_bytes) in vectors {
        let expected = expected_bytes
            .into_iter()
            .map(|byte| byte as usize + 3)
            .collect::<Vec<_>>();
        let encoded = tokenizer.encode(text).unwrap();
        assert_eq!(encoded, expected, "golden ids for {text:?}");
        assert_eq!(tokenizer.decode(&encoded).unwrap(), text);
    }
    assert_eq!(
        tokenizer.decode(&[0, 3 + b'A' as usize, 1, 2]).unwrap(),
        "A"
    );
}

#[test]
fn fixed_qwen3_chat_template_vectors_cover_thinking_modes() {
    let messages = [ChatMessage::new("user", "你好\n42!")];
    let enabled = Qwen3ChatTemplate
        .render(&messages, ThinkingMode::Enabled, true)
        .unwrap();
    let disabled = Qwen3ChatTemplate
        .render(&messages, ThinkingMode::Disabled, true)
        .unwrap();
    assert_eq!(
        enabled,
        "<|im_start|>user\n你好\n42!<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        disabled,
        "<|im_start|>user\n你好\n42!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}
