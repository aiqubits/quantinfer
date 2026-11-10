//! Optional C ABI for the QuantInfer CPU runtime.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{ptr, slice};

use quantinfer::cpu::{load_gguf_tokenizer, CpuModel, ModelSource};
use quantinfer::runtime::{DecodeSession, GenerationConfig};
use quantinfer::tokenizer::{GgufBpeTokenizer, Tokenizer};

const EOS_AUTO: u32 = u32::MAX;
const EOS_DISABLED: u32 = u32::MAX - 1;

pub struct Model {
    decoder: CpuModel,
    tokenizer: Option<GgufBpeTokenizer>,
}

#[repr(i32)]
pub enum Status {
    Ok = 0,
    InvalidArgument = 1,
    BufferTooSmall = 2,
    RuntimeError = 3,
}

#[repr(C)]
pub struct GenerationOptions {
    pub context_length: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub eos_token_id: u32,
    pub seed: u64,
}

type ApiResult<T> = Result<T, (Status, String)>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

#[unsafe(no_mangle)]
pub extern "C" fn quantinfer_abi_version() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn quantinfer_last_error() -> *const c_char {
    LAST_ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn quantinfer_generation_defaults() -> GenerationOptions {
    GenerationOptions {
        context_length: 0,
        max_new_tokens: 32,
        temperature: 0.0,
        top_k: 0,
        top_p: 0.0,
        eos_token_id: EOS_AUTO,
        seed: 0,
    }
}

/// Load a GGUF file or a verified model package.
///
/// # Safety
/// A non-null path must point to a readable NUL-terminated string. A non-null
/// output pointer must be writable for one model pointer and must not alias the path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quantinfer_model_open(
    path: *const c_char,
    out_model: *mut *mut Model,
) -> Status {
    boundary(|| {
        if out_model.is_null() {
            return Err(argument("model output pointer is null"));
        }
        // SAFETY: The caller supplies writable storage for a model pointer.
        unsafe { out_model.write(ptr::null_mut()) };
        // SAFETY: The caller supplies the path as a NUL-terminated string.
        let path = unsafe { utf8(path) }?;
        let source = ModelSource::open(path).map_err(runtime)?;
        let reader = source.open_reader(None).map_err(runtime)?;
        let manifest = source.package().map(|package| &package.manifest);
        let decoder = CpuModel::from_gguf(&reader, manifest, None).map_err(runtime)?;
        let tokenizer = reader
            .metadata()
            .contains_key("tokenizer.ggml.tokens")
            .then(|| load_gguf_tokenizer(&reader))
            .transpose()
            .map_err(runtime)?;
        let handle = Box::into_raw(Box::new(Model { decoder, tokenizer }));
        // SAFETY: The output pointer was checked and is exclusively owned by the caller.
        unsafe { out_model.write(handle) };
        Ok(())
    })
}

/// Release a model obtained from `quantinfer_model_open`.
///
/// # Safety
/// A non-null handle must come from this library and must be released only once,
/// after all operations using it have returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quantinfer_model_free(model: *mut Model) {
    if !model.is_null() {
        // SAFETY: The caller transfers the original boxed handle back exactly once.
        drop(unsafe { Box::from_raw(model) });
    }
}

/// Encode UTF-8 text to tokens. The written count is returned even if the buffer is too small.
///
/// # Safety
/// The handle must remain valid for the call. A non-null text pointer must address
/// a NUL-terminated string. Output buffers and their count must be writable,
/// aligned and disjoint from inputs and each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quantinfer_tokenize(
    model: *const Model,
    text: *const c_char,
    output: *mut u32,
    capacity: usize,
    written: *mut usize,
) -> Status {
    boundary(|| {
        // SAFETY: The C caller supplies a live model and readable string.
        let tokenizer = unsafe { handle(model) }?
            .tokenizer
            .as_ref()
            .ok_or_else(missing_tokenizer)?;
        let tokens = tokenizer.encode(unsafe { utf8(text) }?).map_err(runtime)?;
        let tokens = to_c_tokens(tokens)?;
        // SAFETY: The caller supplies disjoint output storage as described above.
        unsafe { write_output(&tokens, output, capacity, written) }
    })
}

/// Decode tokens to UTF-8. The written byte count includes the terminating NUL.
///
/// # Safety
/// The handle must remain valid. Nonempty input must address `count` initialized
/// tokens. Output storage and its count must be writable and disjoint from inputs
/// and each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quantinfer_detokenize(
    model: *const Model,
    tokens: *const u32,
    count: usize,
    output: *mut c_char,
    capacity: usize,
    written: *mut usize,
) -> Status {
    boundary(|| {
        // SAFETY: The C caller supplies a live model and readable token input.
        let model = unsafe { handle(model) }?;
        let tokens = unsafe { input_tokens(tokens, count) }?;
        let tokenizer = model.tokenizer.as_ref().ok_or_else(missing_tokenizer)?;
        let mut text = tokenizer.decode(&tokens).map_err(runtime)?.into_bytes();
        text.push(0);
        // SAFETY: The caller supplies disjoint output storage as described above.
        unsafe { write_output(&text, output.cast(), capacity, written) }
    })
}

/// Generate token IDs with a fresh KV cache and no model mutation.
///
/// # Safety
/// The handle must remain valid. Nonempty input must address `count` initialized
/// tokens. If provided, options must be readable. Output storage and its count
/// must be writable, aligned and disjoint from inputs and each other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quantinfer_generate_tokens(
    model: *const Model,
    tokens: *const u32,
    count: usize,
    options: *const GenerationOptions,
    output: *mut u32,
    capacity: usize,
    written: *mut usize,
) -> Status {
    boundary(|| {
        if written.is_null() {
            return Err(argument("written count pointer is null"));
        }
        // SAFETY: The caller supplies a writable count and the input storage in the contract.
        unsafe { written.write(0) };
        let model = unsafe { handle(model) }?;
        let tokens = unsafe { input_tokens(tokens, count) }?;
        if tokens.is_empty() {
            return Err(argument("generation requires a nonempty prompt"));
        }
        let defaults = quantinfer_generation_defaults();
        let options = unsafe { options.as_ref() }.unwrap_or(&defaults);
        let eos_token_id = match options.eos_token_id {
            EOS_AUTO => model.tokenizer.as_ref().and_then(Tokenizer::eos_token_id),
            EOS_DISABLED => None,
            id => Some(id as usize),
        };
        if tokens
            .iter()
            .chain(eos_token_id.iter())
            .any(|&id| id >= model.decoder.vocab_size())
        {
            return Err(argument("token ID is outside the model vocabulary"));
        }
        let config = GenerationConfig {
            max_new_tokens: options.max_new_tokens,
            temperature: options.temperature,
            top_k: (options.top_k > 0).then_some(options.top_k),
            top_p: (options.top_p != 0.0).then_some(options.top_p),
            eos_token_id,
            seed: options.seed,
        };
        config
            .validate()
            .map_err(|error| argument(error.to_string()))?;
        let requested = count
            .checked_add(config.max_new_tokens)
            .ok_or_else(|| argument("generation context overflows size_t"))?;
        let context = if options.context_length == 0 {
            requested
        } else {
            options.context_length
        };
        if requested > context || context > model.decoder.max_position_embeddings() {
            return Err(argument("prompt and output exceed the model context"));
        }
        if capacity < config.max_new_tokens {
            // SAFETY: The output count storage was checked above.
            unsafe { written.write(config.max_new_tokens) };
            return Err(buffer_size(config.max_new_tokens));
        }
        if output.is_null() {
            return Err(argument("token output pointer is null"));
        }
        let cache = model.decoder.new_cache(context).map_err(runtime)?;
        let mut session =
            DecodeSession::new(&model.decoder, cache, tokens, config).map_err(runtime)?;
        let mut generated = Vec::new();
        while let Some(token) = session.step().map_err(runtime)? {
            generated.push(u32::try_from(token).map_err(runtime)?);
        }
        // SAFETY: The caller supplies disjoint output storage as described above.
        unsafe { write_output(&generated, output, capacity, written) }
    })
}

fn boundary(run: impl FnOnce() -> ApiResult<()>) -> Status {
    let (status, message) = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(())) => return Status::Ok,
        Ok(Err(error)) => error,
        Err(_) => runtime("QuantInfer operation panicked"),
    };
    let error = CString::new(message.replace('\0', "\\0")).expect("NUL bytes were replaced");
    LAST_ERROR.with(|last| *last.borrow_mut() = error);
    status
}

fn argument(message: impl Into<String>) -> (Status, String) {
    (Status::InvalidArgument, message.into())
}

fn runtime(error: impl std::fmt::Display) -> (Status, String) {
    (Status::RuntimeError, error.to_string())
}

fn missing_tokenizer() -> (Status, String) {
    runtime("model has no embedded tokenizer")
}

fn buffer_size(required: usize) -> (Status, String) {
    (
        Status::BufferTooSmall,
        format!("output requires {required} elements"),
    )
}

unsafe fn handle<'a>(model: *const Model) -> ApiResult<&'a Model> {
    // SAFETY: Non-null model pointers must point to a live handle for the entire call.
    unsafe { model.as_ref() }.ok_or_else(|| argument("model pointer is null"))
}

unsafe fn utf8<'a>(text: *const c_char) -> ApiResult<&'a str> {
    if text.is_null() {
        return Err(argument("UTF-8 string pointer is null"));
    }
    // SAFETY: Non-null string pointers must be readable up to and including the first NUL.
    unsafe { CStr::from_ptr(text) }
        .to_str()
        .map_err(|error| argument(error.to_string()))
}

unsafe fn input_tokens(tokens: *const u32, count: usize) -> ApiResult<Vec<usize>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if tokens.is_null() || count > isize::MAX as usize / size_of::<u32>() {
        return Err(argument("invalid token input buffer"));
    }
    // SAFETY: The caller supplies count initialized tokens in one allocation.
    let tokens = unsafe { slice::from_raw_parts(tokens, count) };
    Ok(tokens.iter().map(|&token| token as usize).collect())
}

fn to_c_tokens(tokens: Vec<usize>) -> ApiResult<Vec<u32>> {
    tokens
        .into_iter()
        .map(|token| u32::try_from(token).map_err(runtime))
        .collect()
}

unsafe fn write_output<T: Copy>(
    values: &[T],
    output: *mut T,
    capacity: usize,
    written: *mut usize,
) -> ApiResult<()> {
    if written.is_null() {
        return Err(argument("written count pointer is null"));
    }
    // SAFETY: The caller supplies writable storage for the count.
    unsafe { written.write(values.len()) };
    if capacity < values.len() {
        return Err(buffer_size(values.len()));
    }
    if !values.is_empty() {
        if output.is_null() {
            return Err(argument("output pointer is null"));
        }
        // SAFETY: The caller supplies disjoint writable storage of at least capacity elements.
        unsafe { ptr::copy_nonoverlapping(values.as_ptr(), output, values.len()) };
    }
    Ok(())
}
