# Dependency boundaries

```text
applications and bindings
  -> quantinfer facade
  -> runtime / model / quantization / format / tokenizer
  -> core and kernel contracts
  -> optional hardware backends
```

The public `quantinfer` facade is the composition boundary used by the CLI,
server, C binding, WebAssembly targets, and WASI-NN adapter. It re-exports the
stable crates and only exposes hardware crates behind feature and target
guards.

- `quantinfer-core` contains tensor, shape, device, storage, and logical
  quantization types. It must not depend on a vendor SDK, HTTP server, model
  architecture, or deployment adapter.
- `quantinfer-kernel` defines operator and backend contracts. Hardware crates
  implement those contracts without changing core or runtime semantics.
- `quantinfer-format` reads and writes generic tensor archives, manifests, and
  QIF. It does not depend on model architecture modules.
- `quantinfer-model` defines architecture configuration and semantic adapters.
  Its `ModelSpec` contains semantic tensor roles and logical shapes only; it
  does not depend on GGUF, SafeTensors, or canonical storage names.
- `quantinfer-quant` is an offline pipeline over core tensors. Runtime code may
  consume its provider contracts, but model and format crates do not depend on
  a particular quantization algorithm.
- `quantinfer-runtime` owns execution and generation. It may combine model,
  format, kernel, core, and quantization contracts, but hardware selection
  remains in provider/back-end boundaries. GGUF storage-name resolution is
  isolated in the runtime format boundary and is not part of model semantics.
- `adapters/` are optional checkpoint or external-engine integrations. Their
  types are converted before entering core, kernel, or runtime APIs.
- `tools/quantinfer-convert` and `tools/xtask` are build-time and offline
  workflows. They may depend on several crates but are never runtime
  dependencies.

The repository does not preserve empty module directories as dependency
claims. A new boundary is added when a real implementation or independent
dependency requires it.
