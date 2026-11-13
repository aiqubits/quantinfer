//! In-memory CPU inference for WebAssembly hosts.

#![forbid(unsafe_code)]

use quantinfer::cpu::{load_gguf_tokenizer, CpuModel};
use quantinfer::format::PackedTensorReader;
use quantinfer::runtime::{DecodeSession, GenerationConfig, Result, RuntimeError};
use quantinfer::tokenizer::{GgufBpeTokenizer, Tokenizer};

pub struct WasmModel {
    decoder: CpuModel,
    tokenizer: Option<GgufBpeTokenizer>,
}

const MAX_GENERATION_TOKENS: usize = 4096;

impl WasmModel {
    /// Take ownership of GGUF or QIF file bytes supplied by the host.
    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        let reader = PackedTensorReader::open_artifact(bytes)?;
        let decoder = CpuModel::from_gguf(&reader, None, None)?;
        let tokenizer = reader
            .metadata()
            .contains_key("tokenizer.ggml.tokens")
            .then(|| load_gguf_tokenizer(&reader))
            .transpose()?;
        Ok(Self { decoder, tokenizer })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<usize>> {
        self.tokenizer()?
            .encode(text)
            .map_err(|error| RuntimeError::InvalidInput(error.to_string()))
    }

    pub fn detokenize(&self, tokens: &[usize]) -> Result<String> {
        self.tokenizer()?
            .decode(tokens)
            .map_err(|error| RuntimeError::InvalidInput(error.to_string()))
    }

    pub fn generate_tokens(
        &self,
        tokens: Vec<usize>,
        config: GenerationConfig,
    ) -> Result<Vec<usize>> {
        config.validate()?;
        if config.max_new_tokens > MAX_GENERATION_TOKENS {
            return Err(RuntimeError::InvalidInput(format!(
                "max_new_tokens must be <= {MAX_GENERATION_TOKENS}"
            )));
        }
        if tokens.is_empty()
            || tokens
                .iter()
                .chain(config.eos_token_id.iter())
                .any(|&token| token >= self.decoder.vocab_size())
        {
            return Err(RuntimeError::InvalidInput(
                "generation requires a nonempty prompt and valid token IDs".to_owned(),
            ));
        }
        let context = tokens
            .len()
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| {
                RuntimeError::InvalidInput("generation context overflows usize".to_owned())
            })?;
        let cache = self.decoder.new_cache(context)?;
        let mut session = DecodeSession::new(&self.decoder, cache, tokens, config)?;
        let mut generated = Vec::new();
        while let Some(token) = session.step()? {
            generated.push(token);
        }
        Ok(generated)
    }

    pub fn eos_token_id(&self) -> Option<usize> {
        self.tokenizer.as_ref().and_then(Tokenizer::eos_token_id)
    }

    fn tokenizer(&self) -> Result<&GgufBpeTokenizer> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| RuntimeError::InvalidInput("model has no embedded tokenizer".to_owned()))
    }
}
