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

To regenerate the golden file, use the verified model directory with
Transformers 4.57.6 and CPU PyTorch 2.11.0. The reference uses F32 weights,
eager attention, positions starting at zero, and an all-ones attention mask.

```sh
python tests/conformance/smollm2_reference.py /path/to/verified/model > tests/golden/smollm2-135m-f32.json
QUANTINFER_SMOLLM2_GGUF=/path/to/converted/model.gguf \
  cargo test --release --locked -p quantinfer-runtime --test smollm2_conformance -- --ignored --nocapture
```
