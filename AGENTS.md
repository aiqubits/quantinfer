# Agent guide

This file restates the repository rules an agent needs before editing.
`docs/developer-guide/README.md` is the source of those rules. If the two
disagree, follow the developer guide and update this file in the same change.

## Boundaries

Dependency direction stays one-way:

```text
applications and bindings
  -> quantinfer facade
  -> runtime / quantization / model / format
  -> core / kernel contracts
  -> hardware backends
```

`quantinfer-core`, `quantinfer-kernel`, and the scalar CPU backend must not
depend on vendor SDKs, HTTP servers, Candle, or WASI-NN. Format crates must
not depend on model architecture modules. WGPU, CUDA, Metal, Ascend, WASM,
and WASI-NN stay optional. QIF stays experimental until its specification and
conformance tests stabilize. Qwen3 MoE and sliding-window execution are
unsupported.

Format readers return validated storage descriptors. Model adapters create
semantic roles. Quantization providers own inspect, validate, quantize,
dequantize, and export layout. Runtime providers resolve capabilities. Kernel
backends perform numerical operations. Do not infer a capability from a
filename or silently re-quantize a tensor while converting formats.

Keep `unsafe` in optional backend crates. `quantinfer-core`,
`quantinfer-kernel`, and the scalar CPU backend remain safe Rust.
`quantinfer-map` is the only crate that maps model files. The format crate
is its only caller.

## Quantized layout

A new quantized layout is done only when all of these land together:

- a storage descriptor
- validation errors for malformed input
- a synthetic round trip
- a runtime dispatch test
- a conformance or numerical gate when an external reference exists

## Commits

Commits go directly to `main` in small independently buildable units. Push
one commit at a time and wait for every workflow for that commit. Fix a
failed workflow with a follow-up commit before starting the next feature. Do
not rewrite pushed history. A provider change also updates its path-filtered
workflow and its capability documentation.

## Local checks

Run these before pushing. They match the fast CI job:

```sh
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --workspace --locked
cargo run --locked -p quantinfer-xtask -- bench
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Also run `git diff --check`. Model conformance downloads pinned artifacts, so
run it only when the change touches that path:

```sh
cargo run --locked -p quantinfer-xtask -- conformance
```
