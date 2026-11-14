# Directory layout

This document describes the implemented workspace layout. A directory is kept
in the repository when it contains source code, test fixtures, generated
baselines, or a README that defines a real artifact boundary. Empty source
directories are not represented by `.gitkeep`; the directory is added when its
first implementation file lands.

## Stable crate boundaries

- `crates/quantinfer` is the public facade. It re-exports the core, format,
  model, kernel, quantization, runtime, and tokenizer crates and exposes
  feature-gated backends through the same API.
- `crates/quantinfer-core` owns backend-independent types. Its implementation
  is split into `error.rs`, `tensor.rs`, and `quantized.rs`.
- `crates/quantinfer-kernel` owns operator and backend contracts.
- `crates/quantinfer-format` owns GGUF, SafeTensors, manifests, and QIF. The
  facade is in `src/lib.rs`; GGUF and SafeTensors are directory modules split
  by reader, writer/decoder, metadata, and test responsibilities. Manifests
  remain in `manifest.rs`, and QIF is in `qif/mod.rs`.
- `crates/quantinfer-model` owns model configuration and semantic adapters.
  `ModelSpec` keeps architecture semantics separate from its storage-name map;
  format readers do not define model roles.
- `crates/quantinfer-quant` owns offline quantization. Its public facade
  delegates to `types.rs`, `schemes.rs`, `providers.rs`, and
  `imatrix.rs`.
- `crates/quantinfer-runtime` owns model execution, batching, scheduling,
  sessions, KV state, profiles, and generation. Shared linear, GGML-layout,
  and attention helpers live in `linear.rs`, `ggml.rs`, and `attention.rs`;
  Llama and Qwen3 execution remain separate modules.
- `crates/quantinfer-tokenizer` owns tokenizer boundaries and chat templates.

## Applications and integrations

- `apps/quantinfer-cli` contains the executable entry point plus
  `backend.rs`, `calibration.rs`, and command modules under
  `src/commands/`.
- `apps/quantinfer-server` separates configuration, shared state, HTTP routes,
  rate limiting, registry, and the decode engine under `src/`.
- `backends/` contains CPU, SIMD, WGPU, CUDA, Metal, Ascend, T-MAC, and KleidiAI
  implementations. Shader and kernel directories are retained only where
  source artifacts exist.
- `adapters/` contains optional checkpoint and interoperability crates.
- `bindings/` and `targets/` contain the C, WebAssembly, and WASI-NN
  integration crates.
- `tools/` contains conversion and repository-task executables; they are not
  runtime dependencies. `quantinfer-convert` is split into source loading,
  model adapters, quantization policy, tokenizer metadata, conversion
  orchestration, and shared parsing helpers.

## Non-code artifacts

- `docs/` is the design and user documentation tree.
- `configs/`, `examples/`, `model-zoo/`, `scripts/`, `benches/`, and
  `tests/` are retained because their parent README or checked-in fixtures
  define a real workflow. Fixture and conformance subdirectories contain
  scripts or baselines rather than placeholders.

The workspace intentionally does not keep the former empty scaffolding under
`src/`, `backends/`, `tests/`, `examples/`, or `configs/`. New code
should be placed in the narrowest existing crate or module boundary; a new
crate is justified when the code has an independent dependency or lifecycle
boundary.
