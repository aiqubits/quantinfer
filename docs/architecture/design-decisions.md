# Design decisions

1. Native Rust is the primary execution target.
2. The original MVP boundary is Qwen3 Dense on a native CPU scalar backend;
   the current workspace also has a Llama Dense runtime path and optional
   accelerator backends.
3. Tensor, quantized storage, kernel contracts, KV cache, scheduling, and
   generation remain QuantInfer-owned abstractions.
4. SafeTensors, GGUF, QIF, serialization, tokenizer, and numeric conversion
   libraries are boundary dependencies. Their external types are converted at
   format or adapter boundaries.
5. WGPU, CUDA, Metal, Ascend, T-MAC, KleidiAI, and mock implementations are
   backends, not definitions of the core runtime.
6. WASI-NN is a deployment adapter, not a tensor library.
7. Candle is an optional interoperability adapter, not a core dependency.
8. GGUF and SafeTensors are compatibility formats. QIF v1 is implemented with
   round-trip, malformed-input, alignment, and legacy-import coverage; format
   changes must update its conformance tests and documentation.
9. Quantization schemes are selected through logical core descriptors and
   provider contracts. Current native paths include Q8_0, Q4_0, Q4_K, Q5_K,
   Q6_K, mixed-precision plans, and imatrix statistics.
10. Cross-device numerical tolerance is tested explicitly; bitwise identity
    across all GPU implementations is not assumed.
11. Code is split at real ownership boundaries. Empty placeholder directories
    are removed rather than treated as future architecture.
