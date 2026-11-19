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
fields. The RoPE dimension, value head width and scaling checks apply to both
native and GGML dimension profiles; storage order does not alter model semantics.
Llama Dense is also represented by a separate model/runtime module.
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

The format layer supports GGUF, SafeTensors, package manifests, and the current
QIF path. Retired QIF v1 code is limited to internal historical fixtures and is
not accepted by the default runtime. The current QIF wire implementation is
isolated in the private `quantinfer-format::qif_v2` boundary. The format API
supports SINGLE/PACKAGE, independent compression and file-backed reads; the
default model loader opens native SINGLE/PACKAGE controls and reads tensors
on demand, without retaining raw weight bytes during typed decoding. Model
construction still materializes the required Core weights. Independent C/Python readers
cover the original unsigned/uncompressed F32 golden profile and a bounded
core-profile matrix with eight encodings, two layouts, three codecs, aliases,
PACKAGE and SINGLE signatures. See the
[conformance matrix](../formats/qif-conformance.md) for the tested boundaries.

## Remaining plan boundaries

The GPTQ adapter has a full-Hessian calibrated matrix producer for native signed Q4_0/Q8_0,
with independent upstream vectors and a real SmolLM2 projection quality test.
The optional converter `gptq` workflow now calibrates and replaces every dense
projection, checks optional holdout gates and publishes verified QIF with exact
reload parity. Its real whole-model case is the fixed SmolLM2 four-bit profile;
the separate model CI recomputes all 210 projections. The native AWQ workflow
adds activation-aware scaling and clipping for full-model QIF publication.
AutoRound also has a calibrated, per-projection SignRound workflow, with an
independent Torch reference and the same verified whole-model publication
protocol. LLM Compressor has
the WNA16 import adapter requested by the general plan. See the
[GPTQ](../../adapters/gptq/README.md), [AWQ](../../adapters/awq/README.md) and
[AutoRound](../../adapters/autoround/README.md) for their tested profiles.
Ascend now has a concrete CANN F32
MatMul executor and an optional CLI path, with independent C ABI contract tests;
real-device acceptance remains separate. KleidiAI has a pinned native AArch64
Q4_0 output-row microkernel with explicit activation-Q8 error checks and CPU
fallbacks. The T-MAC boundary provides a native Rust LUT4 profile with AVX2/NEON
selectors plus its downstream-kernel hook; it does not bundle the upstream
TVM/autotuning stack. [Studio](../../apps/quantinfer-server/README.md) provides
local calibration, quantization, QIF packaging, profile and registry management,
LoRA registration and model benchmark charts. Its fixed SmolLM2 workflow checks
artifact integrity, task cancellation and restart persistence.

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
