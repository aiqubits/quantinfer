# External adapters

Adapters connect QuantInfer to external Rust inference ecosystems without making them core dependencies.

## Candle

Enable the `candle` feature on `quantinfer` to use `quantinfer::candle`.
`to_candle` and `from_candle` copy floating tensors while preserving shape
and element order. Imports from Candle use F32 storage in QuantInfer.

`packed_from_qif` loads validated Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K payloads
into Candle without dequantizing them. Packed blocks must fit within the
last dimension. The resulting Candle tensor retains the stored row order.

## PEFT LoRA

Pass the directory containing `adapter_config.json` and
`adapter_model.safetensors` to `generate` with a matching Qwen3 or Llama
base model. The runtime applies the adapter to the requested projections
during generation and keeps packed base weights in their existing layout.

```sh
cargo run --release -p quantinfer-cli -- generate \
  --input MODEL.gguf --lora ADAPTER_DIR --prompt "Hello" --max-new-tokens 32
```

The loader accepts standard LoRA and rank-stabilized LoRA on decoder linear
projections and the LM head, including per-module rank and alpha overrides.
It checks projection names and shapes against the base model. Select the
same base checkpoint used for training. When an LM head adapter leaves the
base weights frozen, export with `save_embedding_layers=False` to save only
the A/B factors. Generation with `--lora` currently requires a configuration
without a CPU memory budget.
