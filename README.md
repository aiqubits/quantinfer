# QuantInfer

QuantInfer is a Rust-native edge LLM quantization and inference framework.

The repository is organized around a backend-independent runtime, explicit model formats, an offline quantization pipeline, and optional hardware and deployment adapters.

## Workspace layers

```text
quantinfer facade
  -> runtime / quantization / model / format
  -> core / kernel contracts
  -> CPU / WGPU / CUDA / Metal / Ascend backends
```

The first implementation target is native CPU execution with GGUF and SafeTensors loading. WGPU, QIF, vendor backends, Candle interoperability, and WASI-NN deployment remain optional extensions.

## Checks

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
```
