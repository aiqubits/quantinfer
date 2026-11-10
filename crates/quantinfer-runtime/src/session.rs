//! Request-local state for incremental token generation.

use std::sync::Arc;

use crate::{
    last_logits, select_token, DecodeBatchItem, DecoderModel, GenerationConfig, KvCache, Result,
    RuntimeError,
};

impl<M: DecoderModel + ?Sized> DecoderModel for &M {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<quantinfer_core::Tensor> {
        (**self).forward(token_ids, cache)
    }

    fn forward_batch(
        &self,
        batch: &mut [DecodeBatchItem<'_>],
    ) -> Vec<Result<quantinfer_core::Tensor>> {
        (**self).forward_batch(batch)
    }
}

impl<M: DecoderModel + ?Sized> DecoderModel for Arc<M> {
    fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<quantinfer_core::Tensor> {
        (**self).forward(token_ids, cache)
    }

    fn forward_batch(
        &self,
        batch: &mut [DecodeBatchItem<'_>],
    ) -> Vec<Result<quantinfer_core::Tensor>> {
        (**self).forward_batch(batch)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStopReason {
    Eos,
    MaxTokens,
    Cancelled,
    TimedOut,
    Failed,
}

enum DecodeInput {
    Prompt(Vec<usize>),
    Token(usize),
    Finished(DecodeStopReason),
}

/// Advances one request by one token. KV state is owned by the session.
pub struct DecodeSession<M> {
    model: M,
    cache: KvCache,
    config: GenerationConfig,
    next_input: DecodeInput,
    emitted: usize,
}

impl<M: DecoderModel> DecodeSession<M> {
    pub fn new(
        model: M,
        cache: KvCache,
        prompt_tokens: Vec<usize>,
        config: GenerationConfig,
    ) -> Result<Self> {
        config.validate()?;
        if prompt_tokens.is_empty() {
            return Err(RuntimeError::InvalidInput(
                "decode session requires a prompt".to_owned(),
            ));
        }
        if cache.layers.iter().any(|layer| layer.sequence_length != 0) {
            return Err(RuntimeError::InvalidInput(
                "decode session requires an empty KV cache".to_owned(),
            ));
        }
        if prompt_tokens
            .len()
            .checked_add(config.max_new_tokens)
            .is_none_or(|length| length > cache.max_sequence_length)
        {
            return Err(RuntimeError::InvalidInput(
                "prompt and generation exceed the KV cache capacity".to_owned(),
            ));
        }
        Ok(Self {
            model,
            cache,
            config,
            next_input: DecodeInput::Prompt(prompt_tokens),
            emitted: 0,
        })
    }

    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    pub fn stop_reason(&self) -> Option<DecodeStopReason> {
        match self.next_input {
            DecodeInput::Finished(reason) => Some(reason),
            DecodeInput::Prompt(_) | DecodeInput::Token(_) => None,
        }
    }

    pub fn cancel(&mut self) -> bool {
        if self.stop_reason().is_some() {
            return false;
        }
        self.next_input = DecodeInput::Finished(DecodeStopReason::Cancelled);
        true
    }

    /// Prefill on the first call and decode the previous token on subsequent calls.
    pub fn step(&mut self) -> Result<Option<usize>> {
        if self.stop_reason().is_some() {
            return Ok(None);
        }
        // Any error ends the session because the decoder may have advanced some cache layers.
        let input = std::mem::replace(
            &mut self.next_input,
            DecodeInput::Finished(DecodeStopReason::Failed),
        );
        let logits = match input {
            DecodeInput::Prompt(prompt) => self.model.forward(&prompt, &mut self.cache)?,
            DecodeInput::Token(token) => self.model.forward(&[token], &mut self.cache)?,
            DecodeInput::Finished(_) => unreachable!("the session was active"),
        };
        let token = select_token(&last_logits(&logits)?, &self.config)?;
        self.emitted += 1;
        self.next_input = if self.config.eos_token_id == Some(token) {
            DecodeInput::Finished(DecodeStopReason::Eos)
        } else if self.emitted == self.config.max_new_tokens {
            DecodeInput::Finished(DecodeStopReason::MaxTokens)
        } else {
            DecodeInput::Token(token)
        };
        Ok(Some(token))
    }
}

#[cfg(test)]
mod tests {
    use quantinfer_core::Tensor;

    use super::*;

    struct Increment;

    impl DecoderModel for Increment {
        fn forward(&self, tokens: &[usize], cache: &mut KvCache) -> Result<Tensor> {
            let kv = Tensor::from_f32(vec![tokens.len(), 1, 2], vec![0.0; tokens.len() * 2])?;
            cache.append(0, &kv, &kv)?;
            let token = *tokens.last().unwrap();
            if token >= 8 {
                return Err(RuntimeError::InvalidInput(
                    "fixture token out of range".to_owned(),
                ));
            }
            let mut logits = vec![0.0; 8];
            logits[(token + 1) % 8] = 1.0;
            Ok(Tensor::from_f32(vec![1, 8], logits)?)
        }
    }

    fn session(prompt: Vec<usize>, eos: Option<usize>) -> DecodeSession<Arc<Increment>> {
        DecodeSession::new(
            Arc::new(Increment),
            KvCache::new(1, 8, 1, 2).unwrap(),
            prompt,
            GenerationConfig {
                max_new_tokens: 3,
                eos_token_id: eos,
                ..GenerationConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn advances_isolated_requests_and_stops_on_eos_or_cancellation() {
        let mut first = session(vec![1, 2], Some(4));
        let mut second = session(vec![5], None);
        assert_eq!(first.step().unwrap(), Some(3));
        assert_eq!(second.step().unwrap(), Some(6));
        assert_eq!(first.step().unwrap(), Some(4));
        assert_eq!(first.stop_reason(), Some(DecodeStopReason::Eos));
        assert_eq!(first.cache().sequence_length(0).unwrap(), 3);
        assert_eq!(second.cache().sequence_length(0).unwrap(), 1);
        assert!(second.cancel());
        assert_eq!(second.stop_reason(), Some(DecodeStopReason::Cancelled));
        assert_eq!(second.step().unwrap(), None);
        assert_eq!(first.step().unwrap(), None);
    }

    #[test]
    fn does_not_retry_after_a_decoder_partially_writes_the_cache() {
        let mut failed = session(vec![99], None);
        assert!(failed.step().is_err());
        assert_eq!(failed.cache().sequence_length(0).unwrap(), 1);
        assert_eq!(failed.stop_reason(), Some(DecodeStopReason::Failed));
        assert_eq!(failed.step().unwrap(), None);
    }
}
