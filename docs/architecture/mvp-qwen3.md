# QuantInfer MVP baseline and current status

## Runtime boundary

The MVP uses QuantInfer-owned tensor and kernel contracts. The reference path
is a CPU scalar implementation with F32 activations and quantized linear
weights. It does not require a vendor BLAS, CUDA runtime, llama.cpp, or Candle
at runtime. External crates are used at format and utility boundaries and are
converted before entering `quantinfer-core`, `quantinfer-kernel`, or
`quantinfer-runtime`.

## Model boundary

Qwen3 Dense is the original model target. Its loader validates head dimensions,
grouped-query attention, Q/K head RMSNorm, RoPE configuration,
`tie_word_embeddings`, and unsupported sliding-window or custom RoPE scaling
fields. Llama Dense is also represented by a separate model/runtime module.
MoE and sliding-window execution remain outside the supported scope.

The runtime supports single-sequence prefill/decode, request-local KV caches,
batched scheduling, cancellation and timeout handling, deterministic token
selection, tokenizer-backed prompts, and optional LoRA/profile flows.

## Quantization and format boundary

The historical MVP used Q8_0 with 32 values per block. The current code keeps
that path and adds native Q4_0, Q4_K, Q5_K, and Q6_K providers, mixed-precision
assignments, and imatrix statistics. Linear projection weights can be
quantized while embeddings, normalization, and the language-model head remain
floating point where the model path requires it.

The format layer supports GGUF, SafeTensors, package manifests, and QIF v1.
Readers validate dimensions, duplicate metadata/tensor names, alignment,
checksums, package confinement, and finite floating-point payloads. The QIF
and package behavior is covered by focused conformance tests.

## Repository workflow

Every change is reviewed against the implemented boundaries and must pass
formatting, workspace compilation, tests, and Clippy before it is published.
Hardware-specific vendor measurements remain separate from the portable
workspace gates.

The implementation history progressed through dependency boundaries, core
tensor and kernel types, CPU execution, model loading, quantization, tokenizer
and generation, package/profile support, adapters, scheduling, and release
hardening. The current directory layout reflects implemented ownership rather
than a pre-created list of future modules.
