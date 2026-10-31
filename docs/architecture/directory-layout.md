# Directory layout

This document is the reviewed workspace baseline for QuantInfer.

## Boundaries

- `crates/quantinfer-core` contains backend-independent data types.
- `crates/quantinfer-kernel` defines operator contracts and dispatch.
- `crates/quantinfer-runtime` owns LLM execution, KV cache, scheduling, and generation.
- `crates/quantinfer-quant` owns the offline quantization pipeline.
- `backends/` contains hardware implementations.
- `adapters/candle` is optional external-engine interoperability.
- `targets/wasi-nn` is deployment integration, not the runtime core.
- `model-zoo/` contains metadata only; model weights are external artifacts.

## Initial scope

The first implementation targets Qwen3 Dense, native CPU execution, SafeTensors input, GGUF runtime loading, Q8_0 weight-only quantization, KV cache, and single-sequence generation. Qwen3 MoE, sliding-window execution, GPU backends, QIF, vendor SDKs, WASM, and WASI-NN remain extensions.
