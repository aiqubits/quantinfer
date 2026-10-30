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

The first implementation should target native CPU execution, GGUF/SafeTensors loading, Llama/Qwen model families, and a small quantized-kernel surface. WGPU, QIF, vendor backends, Candle, WASM, and WASI-NN are optional extensions.
