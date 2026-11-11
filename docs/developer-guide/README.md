# Developer guide

The repository is a Cargo virtual workspace. Keep dependency direction
one-way: core tensor contracts do not depend on model, format, runtime, or
vendor crates; optional backends and bindings depend on the public contracts.

## Local checks

Before pushing a commit, run the same checks as the fast CI job:

```sh
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --workspace --locked
cargo run -p quantinfer-xtask -- bench
cargo clippy --workspace --all-targets --locked -- -D warnings
git diff --check
```

Model conformance is opt-in locally because it downloads pinned artifacts:

```sh
cargo run -p quantinfer-xtask -- conformance
```

## Change boundaries

Format readers return validated storage descriptors. Model adapters create
semantic roles. Quantization providers own inspect/validate/quantize/dequantize
and export layout. Runtime providers resolve capabilities. Kernel backends
perform numerical operations. Do not infer a capability from a filename or
silently re-quantize a tensor while converting formats.

Every new quantized layout needs a descriptor, validation errors, a synthetic
round trip, a runtime dispatch test, and a conformance or numerical gate when
an external reference exists. Keep unsafe code in optional backend crates; the
core, scalar CPU backend, and kernel contracts remain safe Rust.

## Commit and CI workflow

Commits go directly to `main` in small independently buildable units. Push one
commit at a time and wait for every workflow for that commit. A failed workflow
is fixed before starting the next feature. Changes touching a provider should
also update its path-filtered workflow and its capability documentation.
