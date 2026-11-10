# Deployment targets

Deployment-specific packaging and host integration for WASM and WASI-NN.

## WebAssembly

`quantinfer-target-wasm` loads GGUF or QIF bytes provided by the host and
runs CPU inference in WebAssembly. The JavaScript package exposes the same
runtime through the generated browser module.

```sh
wasm-pack build bindings/wasm --target web --release --no-opt -- --locked
```

```javascript
import init, { Model } from "./pkg/quantinfer_wasm.js";

await init();
const response = await fetch("./model.gguf");
if (!response.ok) throw new Error(`Model download failed with HTTP ${response.status}`);
const model = new Model(new Uint8Array(await response.arrayBuffer()));
try {
    console.log(model.generate("The capital of France is", 8));
} finally {
    model.free();
}
```

`tokenize`, `detokenize`, and `generateTokens` use `Uint32Array` token
IDs. JavaScript generation is greedy and uses the embedded EOS token
when present. Inference runs on one CPU thread and the model must fit
in the host's WebAssembly memory.

## WASI-NN

Enable `wasmtime-host` on `quantinfer-target-wasi-nn` to register a
`QuantInferBackend` in a Wasmtime application. Component guests select the
`ggml` graph encoding and the CPU execution target with one GGUF or QIF
builder. Hosts can also preload verified model packages by name.

```rust
use quantinfer_target_wasi_nn::QuantInferBackend;
use wasmtime_wasi_nn::{wit::WasiNnCtx, Backend, InMemoryRegistry, Registry};

let mut backend = QuantInferBackend::default();
let mut registry = InMemoryRegistry::new();
registry.load(&mut backend, std::path::Path::new("models/smol"))?;
let nn = WasiNnCtx::new([Backend::from(backend)], Registry::from(registry));
```

Expose the context with `wasmtime_wasi_nn::wit::add_to_linker` and a
`WasiNnView`. Legacy WITX guests use `load_by_name` with a host-preloaded
graph and a context from `wasmtime_wasi_nn::witx`.

| Tensor | Legacy index | Data |
| --- | --- | --- |
| `prompt` | Input 0 | Rank-one little-endian I32 token IDs or a U8 UTF-8 prompt |
| `max_new_tokens` | Input 1 | Optional I32 vector with one element, defaults to 1 |
| `tokens` | Output 0 | Rank-one little-endian I32 continuation IDs |
| `text` | Output 1 | U8 UTF-8 continuation when the model has a tokenizer |

Each execution starts a fresh KV cache and uses greedy decoding with the
model's EOS token when available. A host can set artifact size, prompt
size, context length, and generation limits through `WasiNnLimits`.
