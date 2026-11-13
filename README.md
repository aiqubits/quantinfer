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
    cargo check --workspace --locked
    cargo test --workspace --locked
```

## MVP flow

Convert a Qwen3 SafeTensors checkpoint to the canonical GGUF layout. Linear projection matrices are transposed into the runtime layout, and `--quantize q8_0` stores supported projection weights as Q8_0 blocks.

```bash
cargo run --locked -p quantinfer-convert -- \
  --config config.json \
  --weights model.safetensors \
  --output model.gguf \
  --tokenizer tokenizer.json \
  --quantize q8_0
```

The converter also accepts `model.safetensors.index.json` with sharded weights and can emit a verified model manifest. Q4_0 uses the standard GGML block layout.

```bash
cargo run --locked -p quantinfer-convert -- \
  --config config.json \
  --weights model.safetensors.index.json \
  --output model.gguf \
  --manifest manifest.json \
  --quantize q4_0
```

Inspect the resulting metadata and tensor inventory.

```bash
cargo run --locked -p quantinfer-cli -- inspect-gguf --input model.gguf
```

Run CPU Prefill and Decode from a natural-language prompt. The converter embeds the Qwen3 BPE vocabulary and merges into GGUF.

```bash
cargo run --locked -p quantinfer-cli -- generate \
  --input model.gguf \
  --prompt "Hello, QuantInfer" \
  --chat \
  --max-new-tokens 16 \
  --temperature 0.7 \
  --seed 42
```

The token-id form remains available for runtime smoke tests.

```bash
cargo run --locked -p quantinfer-cli -- generate \
  --input model.gguf \
  --tokens 1,2,3 \
  --max-new-tokens 16 \
  --eos-token-id 2
```

The GGUF reader accepts F32, F16, BF16, Q8_0, Q4_0, and standard Q4_K tensors. Linear quantized weights execute through the CPU reference kernel, while floating point tensors are normalized to F32 at load time. `--temperature` and `--seed` enable deterministic seeded sampling. `ByteTokenizer` remains a deterministic fixture for unit tests.

The CPU runtime also supports Q4_0 linear weights and Llama-style Dense models with GQA. A model package directory containing `manifest.json` and its checksummed `model.gguf` can be passed directly to `quantinfer-cli`.
