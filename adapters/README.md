# External adapters

Adapters connect QuantInfer to external Rust inference ecosystems without making them core dependencies.

## Candle

Enable the `candle` feature on `quantinfer` to use `quantinfer::candle`.
`to_candle` and `from_candle` copy floating tensors while preserving shape
and element order. Imports from Candle use F32 storage in QuantInfer.

`packed_view_from_qif` loads Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K blocks without
requantizing them. It verifies the selected tensor's raw hash/finite values,
supports none/zstd/LZ4 and aliases, and returns `PackedQifTensor`. Its `storage()`
keeps the physical packed rows; `logical_shape()` and `is_transposed()` describe
the coordinate view. `dequantize()` returns the original QIF logical values,
applying Candle transpose strides when needed. Blocks must fit each physical
row. This does not perform model-specific RoPE channel transforms.

The existing `packed_from_qif` returns a plain Candle `QTensor` with QIF logical
shape. It rejects native output-row matrices that require a transpose view;
use `packed_view_from_qif` for those matrices. In the GGML dimension profile,
row-major and output-row bytes can already share native logical row order.
The optional Candle CI compares all five encodings, two layouts, two dimension
profiles, three codecs and aliases on non-square matrices, and retains a
specific regression for the previously misread `[32,64]` output-row matrix.

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

`quantinfer-adapter-gptq` also provides a full-Hessian calibrated GPTQ
producer for static signed four/eight-bit groups. It freezes F16 scales before
error feedback and emits native Q4_0/Q8_0 output-row blocks without a second
quantization. Calibration statistics, options, independent upstream vectors,
the real SmolLM2 projection check and remaining workflow scope are documented
in the [GPTQ provider guide](gptq/README.md).

`quantinfer-adapter-gptq` decodes standard GPTQ SafeTensors linear weights
(`qweight`, `qzeros`, `scales`, and optional `g_idx`) into a validated F32
matrix. It supports 2-, 4-, and 8-bit 32-bit-word packing, grouped scales,
the standard packed-output zero-point orientation, and transposed/direct
zero-point variants. One-file and indexed sharded SafeTensors sources are
accepted; the importer does not silently re-quantize decoded values.

The default zero-point view is `[groups, packed_output]`, including square
tables. Other shapes retain the unambiguous transposed/direct compatibility
views. For an ambiguous nonstandard table, use
`dequantize_with_zero_point_layout` with `ZeroPointLayout::{GroupPacked,
PackedGroup, GroupDirect, DirectGroup}`. An explicit layout must match the
source shape. No shape-only inference can distinguish a square table from
its transpose.

Stored zero codes are restored modulo the code bit width. `zero_offset=1`
is the GPTQ-v1 `(zero - 1)` representation; `zero_offset=0` reads direct
codes. The group-major orientation and v1 wrap follow the pinned
[AutoGPTQ implementation](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py).
The importers reject reconstructed weights outside finite F32 range,
including overflow from a finite source scale. Their decode tests compare
all supported bit widths, the four zero layouts and zero-code boundaries
with independent scalar arithmetic; these checks do not implement GPTQ's
calibration/quantization algorithm.

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

Present settings with invalid types or out-of-range integers are rejected;
defaults apply only when the setting is absent. Integer/boolean aliases must
agree, and every supplied method alias must name a supported method. The
explicit `sym`/`symmetric` settings take precedence over the optional
`zero_point` fallback. `dequantize_with_zero_point_layout` exposes the same
explicit layout choice as the GPTQ importer.
