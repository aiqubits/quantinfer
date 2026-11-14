//! Deterministic byte tokenizer used by fixtures and smoke tests.

use super::{Result, Tokenizer, TokenizerError};

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
