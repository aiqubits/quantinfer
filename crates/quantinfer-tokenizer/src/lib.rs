//! Tokenizer and prompt-template boundaries.

#![forbid(unsafe_code)]

use std::collections::HashMap;
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

/// Byte-level BPE tokenizer assembled from the tokenizer arrays stored in GGUF.
#[derive(Debug, Clone)]
pub struct GgufBpeTokenizer {
    tokens: Vec<String>,
    token_ids: HashMap<String, usize>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    special_tokens: Vec<(String, usize)>,
    bos_token_id: Option<usize>,
    eos_token_id: Option<usize>,
}

impl GgufBpeTokenizer {
    pub fn new(
        tokens: Vec<String>,
        merges: &[String],
        bos_token_id: Option<usize>,
        eos_token_id: Option<usize>,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(TokenizerError::InvalidInput(
                "GGUF tokenizer vocabulary cannot be empty".to_owned(),
            ));
        }
        for token_id in [bos_token_id, eos_token_id].into_iter().flatten() {
            if token_id >= tokens.len() {
                return Err(TokenizerError::InvalidInput(format!(
                    "GGUF tokenizer special token id {token_id} is outside the vocabulary"
                )));
            }
        }

        let mut token_ids = HashMap::with_capacity(tokens.len());
        for (token_id, token) in tokens.iter().enumerate() {
            token_ids.entry(token.clone()).or_insert(token_id);
        }

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let mut parts = merge.split_whitespace();
            let Some(left) = parts.next() else {
                return Err(TokenizerError::InvalidInput(format!(
                    "invalid empty BPE merge at rank {rank}"
                )));
            };
            let Some(right) = parts.next() else {
                return Err(TokenizerError::InvalidInput(format!(
                    "BPE merge at rank {rank} must contain two tokens"
                )));
            };
            if parts.next().is_some() {
                return Err(TokenizerError::InvalidInput(format!(
                    "BPE merge at rank {rank} contains more than two tokens"
                )));
            }
            merge_ranks
                .entry((left.to_owned(), right.to_owned()))
                .or_insert(rank);
        }

        let byte_encoder = byte_encoder();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(byte, character)| (*character, byte as u8))
            .collect();
        let mut special_tokens: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter(|(token_id, token)| {
                is_special_token(token)
                    || Some(*token_id) == bos_token_id
                    || Some(*token_id) == eos_token_id
            })
            .map(|(token_id, token)| (token.to_string(), token_id))
            .collect();
        special_tokens.sort_by(|left, right| right.0.len().cmp(&left.0.len()));

        Ok(Self {
            tokens,
            token_ids,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            special_tokens,
            bos_token_id,
            eos_token_id,
        })
    }

    fn encode_segment(&self, text: &str) -> Result<Vec<usize>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut symbols: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|byte| self.byte_encoder[*byte as usize].to_string())
            .collect();
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len() - 1 {
                let pair = (&symbols[index], &symbols[index + 1]);
                let Some(rank) = self
                    .merge_ranks
                    .get(&(pair.0.clone(), pair.1.clone()))
                    .copied()
                else {
                    continue;
                };
                if best.is_none_or(|(best_rank, _)| rank < best_rank) {
                    best = Some((rank, index));
                }
            }
            let Some((_, index)) = best else {
                break;
            };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols.splice(index..=index + 1, [merged]);
        }

        symbols
            .into_iter()
            .map(|symbol| {
                self.token_ids.get(&symbol).copied().ok_or_else(|| {
                    TokenizerError::InvalidInput(format!(
                        "GGUF tokenizer vocabulary cannot encode byte-level token {symbol:?}"
                    ))
                })
            })
            .collect()
    }
}

impl Tokenizer for GgufBpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut output = Vec::new();
        let mut segment_start = 0;
        let mut cursor = 0;
        while cursor < text.len() {
            let rest = &text[cursor..];
            let special = self
                .special_tokens
                .iter()
                .find(|(token, _)| rest.starts_with(token));
            if let Some((token, token_id)) = special {
                output.extend(self.encode_segment(&text[segment_start..cursor])?);
                output.push(*token_id);
                cursor += token.len();
                segment_start = cursor;
            } else {
                cursor += rest
                    .chars()
                    .next()
                    .expect("cursor is below text length")
                    .len_utf8();
            }
        }
        output.extend(self.encode_segment(&text[segment_start..])?);
        Ok(output)
    }

    fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            let token = self
                .tokens
                .get(*token_id)
                .ok_or(TokenizerError::UnknownToken(*token_id))?;
            if Some(*token_id) == self.bos_token_id
                || Some(*token_id) == self.eos_token_id
                || is_special_token(token)
            {
                continue;
            }
            for character in token.chars() {
                if let Some(byte) = self.byte_decoder.get(&character) {
                    bytes.push(*byte);
                } else {
                    let mut encoded = [0_u8; 4];
                    bytes.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|_| TokenizerError::InvalidInput("decoded tokens are not UTF-8".to_owned()))
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id
    }
}

fn is_special_token(token: &str) -> bool {
    token.starts_with("<|") && token.ends_with("|>")
}

fn byte_encoder() -> [char; 256] {
    let mut bytes = Vec::new();
    bytes.extend(33_u8..=126);
    bytes.extend(161_u8..=172);
    bytes.extend(174_u8..=255);
    let mut codepoints: Vec<u32> = bytes.iter().map(|byte| u32::from(*byte)).collect();
    let mut next_codepoint = 0_u32;
    for byte in 0_u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + next_codepoint);
            next_codepoint += 1;
        }
    }
    let mut encoder = ['\0'; 256];
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        encoder[byte as usize] = char::from_u32(codepoint).expect("byte encoder codepoint");
    }
    encoder
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
    use super::{
        ByteTokenizer, ChatMessage, GgufBpeTokenizer, Qwen3ChatTemplate, ThinkingMode, Tokenizer,
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
        assert!(prompt.ends_with("<|im_start|>assistant\n/no_think\n"));
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
}
