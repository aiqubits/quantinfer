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
5. WGPU, CUDA, Metal, Ascend, T-MAC, and KleidiAI belong to backend boundaries,
   not definitions of the core runtime. External kernel hooks remain available.
   Ascend has a concrete optional CANN F32 executor and CLI entry point; its C
   ABI contract tests do not replace actual NPU acceptance.
   KleidiAI optionally builds a pinned Arm SDK for its Q4_0 output-row path;
   activation Q8 quantization and CPU fallback boundaries are explicit.
   The native Rust T-MAC LUT4 profile uses AVX2/NEON lookup and preserves F32
   activations/scales; it is separate from the upstream TVM-generated stack.
6. WASI-NN is a deployment adapter, not a tensor library.
7. Candle is an optional interoperability adapter, not a core dependency.
8. GGUF and SafeTensors are compatibility formats. QIF v1 is retired and kept
   only as a legacy migration reference. The current QIF wire implementation is
   isolated in the private `qif_v2` boundary. PACKAGE and compression are format
   API capabilities; the default model loader opens native SINGLE/PACKAGE paths
   with control-only initial I/O and per-tensor reads during typed loading.
   Independent cross-language validation covers the fixed golden profile in the
   [conformance matrix](../formats/qif-conformance.md).
9. Quantization schemes are selected through logical core descriptors and
   provider contracts. Current native paths include Q8_0, Q4_0, Q4_K, Q5_K,
   Q6_K, mixed-precision plans, and imatrix statistics.
10. Cross-device numerical tolerance is tested explicitly; bitwise identity
    across all GPU implementations is not assumed.
11. Code is split at real ownership boundaries. Empty placeholder directories
    are removed rather than treated as future architecture.
