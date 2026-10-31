# Design decisions

1. Native Rust is the primary execution target.
2. The first MVP targets Qwen3 Dense on a native CPU scalar backend.
3. Tensor, quantized storage, kernel contracts, KV cache, and generation remain QuantInfer-owned abstractions.
4. SafeTensors, GGUF, serialization, tokenizer, and numeric conversion libraries are boundary dependencies only.
5. WGPU, CUDA, Metal, and Ascend are backends, not definitions of the core runtime.
6. WASI-NN is a deployment adapter, not a tensor library.
7. Candle is an optional interoperability adapter, not a core dependency.
8. GGUF and SafeTensors are compatibility formats; QIF remains experimental until its specification and conformance tests stabilize.
9. Cross-device numerical tolerance is tested explicitly; bitwise identity across all GPU implementations is not assumed.
