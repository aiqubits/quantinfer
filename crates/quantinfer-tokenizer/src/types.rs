//! Shared tokenizer and chat-template contracts.

use std::fmt;

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

pub type Result<T> = std::result::Result<T, TokenizerError>;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>>;
    fn decode(&self, token_ids: &[usize]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<usize>;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTemplateOptions {
    pub thinking: ThinkingMode,
    pub add_generation_prompt: bool,
    pub add_bos_token: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            thinking: ThinkingMode::Disabled,
            add_generation_prompt: true,
            add_bos_token: false,
        }
    }
}

pub trait ChatTemplate {
    fn render_messages(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String>;
}
