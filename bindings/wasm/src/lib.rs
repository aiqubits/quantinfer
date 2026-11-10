//! JavaScript bindings for in-memory QuantInfer inference.

#![forbid(unsafe_code)]

use quantinfer::runtime::{GenerationConfig, Result as RuntimeResult};
use quantinfer_target_wasm::WasmModel;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model {
    inner: WasmModel,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>) -> Result<Model, JsError> {
        Ok(Self {
            inner: into_js(WasmModel::open(bytes))?,
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, JsError> {
        token_ids(into_js(self.inner.tokenize(text))?)
    }

    pub fn detokenize(&self, tokens: Vec<u32>) -> Result<String, JsError> {
        into_js(self.inner.detokenize(&host_tokens(tokens)))
    }

    #[wasm_bindgen(js_name = generateTokens)]
    pub fn generate_tokens(
        &self,
        tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, JsError> {
        token_ids(self.generate_inner(host_tokens(tokens), max_new_tokens)?)
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String, JsError> {
        let prompt = into_js(self.inner.tokenize(prompt))?;
        let output = self.generate_inner(prompt, max_new_tokens)?;
        into_js(self.inner.detokenize(&output))
    }
}

impl Model {
    fn generate_inner(
        &self,
        tokens: Vec<usize>,
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, JsError> {
        let config = GenerationConfig {
            max_new_tokens,
            eos_token_id: self.inner.eos_token_id(),
            ..GenerationConfig::default()
        };
        into_js(self.inner.generate_tokens(tokens, config))
    }
}

fn host_tokens(tokens: Vec<u32>) -> Vec<usize> {
    tokens.into_iter().map(|token| token as usize).collect()
}

fn token_ids(tokens: Vec<usize>) -> Result<Vec<u32>, JsError> {
    tokens
        .into_iter()
        .map(|token| u32::try_from(token).map_err(|error| JsError::new(&error.to_string())))
        .collect()
}

fn into_js<T>(result: RuntimeResult<T>) -> Result<T, JsError> {
    result.map_err(|error| JsError::new(&error.to_string()))
}
