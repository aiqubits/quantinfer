//! GGUF byte-level BPE tokenizer.

use std::collections::HashMap;

use super::{Result, Tokenizer, TokenizerError};

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
    pre_tokenizer: GgufPreTokenizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufPreTokenizer {
    ByteLevel,
}

impl GgufBpeTokenizer {
    pub fn new(
        tokens: Vec<String>,
        merges: &[String],
        bos_token_id: Option<usize>,
        eos_token_id: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_pre(tokens, merges, bos_token_id, eos_token_id, None)
    }

    /// Build a GGUF BPE tokenizer while honoring the converter's pre-tokenizer
    /// declaration. The current runtime implements the GPT-2 byte-level family.
    pub fn new_with_pre(
        tokens: Vec<String>,
        merges: &[String],
        bos_token_id: Option<usize>,
        eos_token_id: Option<usize>,
        pre_tokenizer: Option<&str>,
    ) -> Result<Self> {
        let pre_tokenizer = match pre_tokenizer.unwrap_or("default") {
            "default" | "gpt2" | "qwen2" | "qwen2new" | "smollm" => GgufPreTokenizer::ByteLevel,
            value => {
                return Err(TokenizerError::InvalidInput(format!(
                    "unsupported GGUF pre-tokenizer {value:?}"
                )))
            }
        };
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
        special_tokens.sort_by_key(|left| std::cmp::Reverse(left.0.len()));

        Ok(Self {
            tokens,
            token_ids,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            special_tokens,
            bos_token_id,
            eos_token_id,
            pre_tokenizer,
        })
    }

    fn encode_segment(&self, text: &str) -> Result<Vec<usize>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoded = Vec::new();
        let segments = match self.pre_tokenizer {
            GgufPreTokenizer::ByteLevel => byte_level_segments(text)?,
        };
        for segment in segments {
            encoded.extend(self.encode_piece(segment)?);
        }
        Ok(encoded)
    }

    fn encode_piece(&self, text: &str) -> Result<Vec<usize>> {
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

fn byte_level_segments(text: &str) -> Result<Vec<&str>> {
    let mut segments = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let start = cursor;
        let rest = &text[cursor..];
        if let Some(length) = contraction_length(rest) {
            cursor += length;
            segments.push(&text[start..cursor]);
            continue;
        }
        let character = rest.chars().next().expect("cursor is below text length");
        let next = rest[character.len_utf8()..].chars().next();
        if character.is_alphabetic()
            || (!character.is_alphanumeric()
                && character != '\r'
                && character != '\n'
                && next.is_some_and(char::is_alphabetic))
        {
            if !character.is_alphabetic() {
                cursor += character.len_utf8();
            }
            consume_while(text, &mut cursor, char::is_alphabetic);
        } else if character.is_numeric() {
            cursor += character.len_utf8();
        } else if character == ' ' && next.is_some_and(is_punctuation) {
            cursor += character.len_utf8();
            consume_punctuation(text, &mut cursor);
        } else if is_punctuation(character) {
            consume_punctuation(text, &mut cursor);
        } else if character.is_whitespace() {
            consume_whitespace(text, &mut cursor);
        } else {
            cursor += character.len_utf8();
        }
        segments.push(&text[start..cursor]);
    }
    Ok(segments)
}

fn contraction_length(text: &str) -> Option<usize> {
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        .iter()
        .find_map(|suffix| {
            text.get(..suffix.len())
                .filter(|prefix| prefix.eq_ignore_ascii_case(suffix))
                .map(|_| suffix.len())
        })
}

fn is_punctuation(character: char) -> bool {
    !character.is_alphanumeric() && !character.is_whitespace()
}

fn consume_while(text: &str, cursor: &mut usize, predicate: fn(char) -> bool) {
    while *cursor < text.len() {
        let character = text[*cursor..]
            .chars()
            .next()
            .expect("cursor is below text length");
        if !predicate(character) {
            break;
        }
        *cursor += character.len_utf8();
    }
}

fn consume_punctuation(text: &str, cursor: &mut usize) {
    consume_while(text, cursor, is_punctuation);
    while *cursor < text.len() {
        let character = text[*cursor..]
            .chars()
            .next()
            .expect("cursor is below text length");
        if character != '\r' && character != '\n' {
            break;
        }
        *cursor += character.len_utf8();
    }
}

fn consume_whitespace(text: &str, cursor: &mut usize) {
    let mut scan = *cursor;
    while scan < text.len() {
        let character = text[scan..]
            .chars()
            .next()
            .expect("scan is below text length");
        if !character.is_whitespace() {
            break;
        }
        scan += character.len_utf8();
    }
    let whitespace = &text[*cursor..scan];
    if whitespace
        .chars()
        .any(|character| character == '\r' || character == '\n')
    {
        let mut newline_end = *cursor;
        while newline_end < text.len() {
            let character = text[newline_end..]
                .chars()
                .next()
                .expect("newline scan is below text length");
            if character != '\r' && character != '\n' {
                break;
            }
            newline_end += character.len_utf8();
        }
        if newline_end == *cursor {
            while *cursor < text.len() {
                let character = text[*cursor..]
                    .chars()
                    .next()
                    .expect("cursor is below text length");
                if character != ' ' && character != '\t' {
                    break;
                }
                *cursor += character.len_utf8();
            }
            while *cursor < text.len() {
                let character = text[*cursor..]
                    .chars()
                    .next()
                    .expect("cursor is below text length");
                if character != '\r' && character != '\n' {
                    break;
                }
                *cursor += character.len_utf8();
            }
        } else {
            *cursor = newline_end;
        }
    } else {
        *cursor = scan;
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
                || is_decode_skipped_token(token)
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
    (token.starts_with("<|") && token.ends_with("|>"))
        || matches!(
            token,
            "<think>"
                | "</think>"
                | "<tool_call>"
                | "</tool_call>"
                | "<tool_response>"
                | "</tool_response>"
        )
}

fn is_decode_skipped_token(token: &str) -> bool {
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
