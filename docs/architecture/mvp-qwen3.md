# QuantInfer first MVP

## Runtime boundary

The MVP uses a QuantInfer-owned tensor and kernel contract. The reference backend is a CPU scalar implementation with F32 activations and Q8_0 linear weights. The implementation does not require a vendor BLAS, CUDA runtime, llama.cpp, or Candle at runtime.

External crates may be used at format and utility boundaries. Their types are converted to QuantInfer types before entering `quantinfer-core`, `quantinfer-kernel`, or `quantinfer-runtime`.

## Model boundary

The first model is Qwen3 Dense. The loader validates `head_dim`, grouped-query attention, Q/K head RMSNorm, RoPE configuration, `tie_word_embeddings`, and unsupported sliding-window or custom RoPE scaling fields. MoE and sliding-window execution are outside the first MVP.

The first generation path is single-sequence Prefill and Decode with a KV cache. Non-thinking generation is the default. Thinking-mode template and sampling support can be enabled after the deterministic path is validated.

## Quantization boundary

The first quantization format is Q8_0 with 32 values per block. Linear projection weights are quantized; RMSNorm, embeddings, and the language-model head remain floating point. The GGUF reader accepts F32, F16, and BF16 floating point tensors and converts them to F32 for the reference runtime.

## Direct-main submission policy

Work is committed directly to `main`; no development branches and no pull requests are used for this MVP. Each commit implements one independently testable capability and is pushed only after the GitHub Actions workflow passes formatting, compilation, tests, and Clippy checks. A failed workflow is fixed with a follow-up commit; pushed history is never rewritten.

The implementation order is:

1. dependency policy and workspace boundaries;
2. core tensor and layout types;
3. CPU scalar kernels and deterministic fixtures;
4. Qwen3 configuration and F32 execution;
5. SafeTensors/GGUF loading and conversion;
6. Q8_0 quantization and quantized execution;
7. tokenizer, Chat Template, and generation;
8. benchmarks and release hardening.

## Current end-to-end path

The CLI generation path loads canonical GGUF directly into `Qwen3Model`, constructs the Qwen3 byte-level BPE tokenizer from GGUF metadata, renders an optional chat prompt, and runs CPU Prefill and Decode. It accepts either a natural-language prompt or comma-separated token IDs. EOS metadata is read from GGUF and can be overridden on the command line. A seed makes temperature sampling reproducible. `ByteTokenizer` remains a deterministic fixture for unit tests.
