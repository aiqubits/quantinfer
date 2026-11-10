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
