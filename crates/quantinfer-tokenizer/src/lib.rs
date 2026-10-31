//! Tokenizer and prompt-template boundaries.

#![forbid(unsafe_code)]

use std::fmt;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    InvalidInput(String),
    UnknownToken(usize),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::UnknownToken(token) => write!(formatter, "unknown token id {token}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>>;
    fn decode(&self, token_ids: &[usize]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<usize>;
}

/// Deterministic byte tokenizer used by fixtures and smoke tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    const BOS: usize = 0;
    const EOS: usize = 1;
    const PAD: usize = 2;
    const BYTE_OFFSET: usize = 3;
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>> {
        Ok(text
            .as_bytes()
            .iter()
            .map(|value| Self::BYTE_OFFSET + usize::from(*value))
            .collect())
    }

    fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            match *token_id {
                Self::BOS | Self::EOS | Self::PAD => continue,
                value if (Self::BYTE_OFFSET..Self::BYTE_OFFSET + 256).contains(&value) => {
                    bytes.push((value - Self::BYTE_OFFSET) as u8)
                }
                value => return Err(TokenizerError::UnknownToken(value)),
            }
        }
        String::from_utf8(bytes)
            .map_err(|_| TokenizerError::InvalidInput("decoded tokens are not UTF-8".to_owned()))
    }

    fn vocab_size(&self) -> usize {
        Self::BYTE_OFFSET + 256
    }

    fn eos_token_id(&self) -> Option<usize> {
        Some(Self::EOS)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3ChatTemplate;

impl Qwen3ChatTemplate {
    pub fn render(
        &self,
        messages: &[ChatMessage],
        thinking: ThinkingMode,
        add_generation_prompt: bool,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(TokenizerError::InvalidInput(
                "chat template requires at least one message".to_owned(),
            ));
        }
        let mut output = String::new();
        for message in messages {
            if message.role.is_empty() || message.content.is_empty() {
                return Err(TokenizerError::InvalidInput(
                    "chat message role and content cannot be empty".to_owned(),
                ));
            }
            output.push_str("<|im_start|>");
            output.push_str(&message.role);
            output.push('\n');
            output.push_str(&message.content);
            output.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            output.push_str("<|im_start|>assistant\n");
            if thinking == ThinkingMode::Disabled {
                output.push_str("/no_think\n");
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{ByteTokenizer, ChatMessage, Qwen3ChatTemplate, ThinkingMode, Tokenizer};

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
        assert!(prompt.ends_with("<|im_start|>assistant\n/no_think\n"));
    }
}
