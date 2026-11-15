//! Synthetic tokenizer and quantization conformance task.

use quantinfer_core::Tensor;
use quantinfer_quant::{quantize_q4_0, quantize_q8_0};
use quantinfer_tokenizer::{
    ByteTokenizer, ChatMessage, Qwen3ChatTemplate, ThinkingMode, Tokenizer,
};

pub(crate) fn run() {
    let tokenizer = ByteTokenizer;
    let prompt = "QuantInfer 中文\n42!";
    let token_ids = tokenizer
        .encode(prompt)
        .expect("fixture tokenizer should encode conformance prompt");
    assert_eq!(tokenizer.decode(&token_ids).unwrap(), prompt);
    let chat_prompt = Qwen3ChatTemplate
        .render(
            &[ChatMessage::new("user", prompt)],
            ThinkingMode::Disabled,
            true,
        )
        .expect("fixture chat template should render");
    assert!(chat_prompt.contains("<|im_start|>assistant"));

    let input = Tensor::from_f32(
        vec![1, 32],
        (0..32).map(|index| index as f32 / 32.0 - 0.5).collect(),
    )
    .expect("fixture input shape is valid");
    let q8 = quantize_q8_0(&input).expect("Q8_0 fixture quantization should succeed");
    let q4 = quantize_q4_0(&input).expect("Q4_0 fixture quantization should succeed");
    let q8_values = quantinfer_quant::dequantize_q8_0(&q8)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let q4_values = quantinfer_quant::dequantize_q4_0(&q4)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let max_error = |values: &[f32]| {
        values
            .iter()
            .zip(input.as_f32().unwrap())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max)
    };
    assert!(max_error(&q8_values) < 0.01);
    assert!(max_error(&q4_values) < 0.08);
    println!(
        "conformance_ok prompt_tokens={} q8_max_error={:.6} q4_max_error={:.6}",
        token_ids.len(),
        max_error(&q8_values),
        max_error(&q4_values)
    );
}
