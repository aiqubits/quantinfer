# Cross-crate tests

Integration, conformance, numerical, compatibility, hardware, fixture, and golden-output tests.

## Qwen3-0.6B F32, Q8_0 and Q4_0 conformance

Model Conformance runs the full model checks on its weekly schedule and on
manual dispatch. It verifies the pinned SafeTensors and tokenizer files,
converts the checkpoint, and checks the F32 and Q8_0 GGUF hashes against
`golden/qwen3-0.6b-baseline.json`.

The test compares every prefill logit against a pinned Transformers CPU
reference and checks weight error, logit error, and top-10 overlap with separate
Q8_0 and Q4_0 thresholds. The fixed non-thinking prompt verifies deterministic
output and EOS for all three variants. F32 and Q8_0 must also match the frozen
token sequence. The workflow publishes the independently generated reference
logits and comparison results.

The tokenizer job also verifies the official `tokenizer_config.json` hash.
Its rendered prompts must exactly match the Transformers 4.57.6 cases in
`golden/qwen3-chat-template.json`, including system and assistant history,
thinking modes, structured tool calls, and consecutive tool responses.

## SmolLM2-135M F32 conformance

The Model Conformance workflow verifies the pinned checkpoint in
`model-zoo/manifests/smollm2-135m.json`, converts it to GGUF, and runs the
runtime integration test. The test compares prompt tokens, sampled prefill
logits, the top 10 tokens, and eight greedy decode tokens with the reference.
The same test runs against the pinned external F16 GGUF from the manifest.
That import converts weights to F32 for CPU execution.

## GGUF interoperability

The GGUF Interoperability workflow uses the llama.cpp revision
`0ec4027a2a0183de87c007069adddc06e665f7de`. It quantizes the pinned external
SmolLM2 F16 artifact to Q8_0 and Q4_0, and also tests QuantInfer's F32, Q8_0
and Q4_0 exports from the pinned SafeTensors checkpoint. The test compares
metadata and all tensor shapes, types and byte lengths against llama.cpp's
GGUF reader, then checks prompt IDs and six greedy generation tokens against
llama-server. Exported models are also loaded through their model packages.

For a BPE-based Llama checkpoint, use the GGML layout option with the
pre-tokenizer name recognized by llama.cpp. SmolLM2 uses `smollm`.

```sh
cargo run --release -p quantinfer-convert -- \
  --config config.json --weights model.safetensors --tokenizer tokenizer.json \
  --gguf-layout ggml --tokenizer-pre smollm --quantize q8_0 --output model.gguf
```

The GGML layout stores linear weights in output rows with adjacent RoPE
pairs for Q/K. QuantInfer keeps the imported Q8_0 and Q4_0 projection blocks
and reorders Q/K rows without requantization. Embedding and output head
weights are decoded to F32 for their existing CPU kernels. A requested
quantization is applied to projections whose input rows are block-aligned.
Omit `--gguf-layout` to produce the original QuantInfer layout.

To regenerate the golden file, use the verified model directory with
Transformers 4.57.6 and CPU PyTorch 2.11.0. The reference uses F32 weights,
eager attention, positions starting at zero, and an all-ones attention mask.

```sh
python tests/conformance/smollm2_reference.py /path/to/verified/model > tests/golden/smollm2-135m-f32.json
QUANTINFER_SMOLLM2_GGUF=/path/to/converted/model.gguf \
  cargo test --release --locked -p quantinfer-runtime --test smollm2_conformance -- --ignored --nocapture
```

## SafeTensors conversion memory

The converter accepts `model.safetensors` or `model.safetensors.index.json`
through `--weights`. It reads and encodes one tensor at a time. GGUF payloads
are spooled to a scratch file in the destination directory, then copied to a
complete GGUF and published atomically. The destination filesystem needs room
for both files during the copy. A requested manifest checksum is calculated
by reading the completed GGUF incrementally.

Use `--weights-checksums source.sha256` to verify the source before writing
any tensors. The SHA-256 list must include the input file basename and, for
an index, every shard path named by that index. Model Conformance checks that
the pinned SmolLM2 checkpoint yields the same GGUF from both input layouts
and that conversion rejects missing or mismatched shard hashes.
