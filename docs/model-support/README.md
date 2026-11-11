# Model support

QuantInfer separates model semantics from file formats. A model adapter reads
architecture configuration and emits `ModelSpec`, tensor roles, attention,
normalization, FFN, positional encoding, vocabulary, and LM-head semantics.
SafeTensors, sharded SafeTensors, GGUF, and QIF readers then map storage into
those roles.

## Supported dense models

### Qwen3 Dense

The Qwen3 adapter validates grouped-query attention, head dimension, Q/K
normalization, RoPE, vocabulary, tied embeddings, and the supported
`Qwen3ForCausalLM` architecture. It supports F32, Q8_0, Q4_0, Q4_K, Q5_K,
Q6_K, KV cache, deterministic generation, official chat templates, and
optional PEFT LoRA projections. Sliding-window attention and custom RoPE
scaling are rejected explicitly until their execution semantics are added.

### Llama Dense

The Llama adapter covers the decoder-only dense layout used by the pinned
SmolLM2-135M fixture, including GQA, RoPE, RMSNorm, F32/Q8_0/Q4_0 execution,
KV cache, and tokenizer metadata. Its independent Transformers reference is
exercised in model conformance CI.

## Quantization and adapters

Native GGUF execution supports Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K. Calibration
statistics and imatrix channel weights are collected through the CLI and are
attached to conversion metadata. Mixed-precision policies are validated against
the model manifest and memory plan.

External SafeTensors adapters cover GPTQ, AWQ, AutoRound, and LLM Compressor
W4A16 layouts. PEFT LoRA adapters are loaded from `adapter_config.json` and
SafeTensors factors and can be attached to loaded dense models or registry
entries.

## Adding a model

1. Add a semantic adapter under `crates/quantinfer-model/src`.
2. Keep source tensor names and format parsing in the adapter/reader boundary.
3. Add a small synthetic fixture for shapes and tensor roles.
4. Add a pinned external artifact, tokenizer vectors, and independent logits
   or generation references under `tests/conformance`.
5. Register the model in the conformance workflow and document unsupported
   configuration fields with an explicit error.

Weights are never committed to the repository. CI downloads a fixed revision,
verifies its checksum, and records the exact artifact in the job summary.
