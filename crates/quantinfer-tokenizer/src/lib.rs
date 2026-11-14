//! Tokenizer and prompt-template boundaries.

#![forbid(unsafe_code)]

mod bpe;
mod byte;
mod chat_template;
mod qwen_chat;
mod types;

pub use bpe::GgufBpeTokenizer;
pub use byte::ByteTokenizer;
pub use chat_template::HuggingFaceChatTemplate;
pub use qwen_chat::Qwen3ChatTemplate;
pub use types::{
    ChatMessage, ChatTemplate, ChatTemplateOptions, Result, ThinkingMode, Tokenizer, TokenizerError,
};

#[cfg(test)]
mod tests;
