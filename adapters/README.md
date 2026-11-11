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

## GPTQ

`quantinfer-adapter-gptq` decodes standard GPTQ SafeTensors linear weights
(`qweight`, `qzeros`, `scales`, and optional `g_idx`) into a validated F32
matrix. It supports 2-, 4-, and 8-bit 32-bit-word packing, grouped scales,
the standard packed-output zero-point orientation, and transposed/direct
zero-point variants. One-file and indexed sharded SafeTensors sources are
accepted; the importer does not silently re-quantize decoded values.

## AWQ

`quantinfer-adapter-awq` decodes AutoAWQ's four-bit output-packed
SafeTensors (`qweight`, `qzeros`, `scales`, and optional `g_idx`) into a
validated F32 matrix. It handles the interleaved AWQ bit order, grouped
scales, symmetric checkpoints without zero tensors, and direct zero-point
fixtures without changing the source's quantization values.

## LLM Compressor

`quantinfer-adapter-llm-compressor` reads the compressed-tensors WNA16
SafeTensors layout (`weight_packed`, `weight_scale`, optional
`weight_zero_point`, and `weight_shape`) and dequantizes it into a validated
F32 matrix. It checks the packed input axis, output-packed zero points, group
scales, optional group indices, and symmetric checkpoints before exposing the
tensor to a runtime.

## AutoRound

`quantinfer-adapter-autoround` parses AutoRound's quantization metadata
aliases and reuses the GPTQ-compatible packed tensor decoder for
`qweight/qzeros/scales` matrices. It accepts one-file and indexed sharded
SafeTensors sources while validating the quantization method, bit width,
group size, and zero-point convention.
