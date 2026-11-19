# Offline calibrated quantization

Build the optional CPU GPTQ workflow with:

```sh
cargo build --release --locked -p quantinfer-convert --features gptq
target/release/quantinfer-convert gptq \
  --input model-f32.gguf --dataset calibration.jsonl \
  --validation-dataset validation.jsonl --maximum-kl 0.5 \
  --minimum-top10-overlap 6 --max-sequence-length 512 \
  --output model.qif --report quantization.json
```

Input can be floating GGUF, native QIF SINGLE/PACKAGE, or a verified model
package with its embedded tokenizer. Qwen3 and Llama dense models are supported.
Every input tensor must be F32/F16/BF16; already-quantized sources fail explicitly.
This entry point does not run AWQ, AutoRound or an uncalibrated RTN fallback.

JSONL datasets contain one JSON string or an object with a `text` string per
line. Nonempty strings may contain whitespace. Blank lines are ignored. Prompt
tokens must fit the configured context (default: model capacity or 512, whichever
is smaller). Each prompt starts with a fresh KV cache. The exact bounded input
snapshot is hashed; the default dataset ceiling is 32 MiB.

The workflow calibrates and replaces all seven projections in each layer,
in Q/K/V, attention-output, gate/up and down groups. Later groups/layers observe
the already-quantized model, and shared inputs are counted once. The
[provider](../../adapters/gptq/README.md) uses the full activation Gram matrix,
relative damping and error feedback with F16 grids fixed before rounding.
Native production supports signed `--bits 4|8`, `--group-size` as a positive
multiple of 32 (default 32), `--damping` (default 0.01), and optional
`--activation-order`. It is a static-group profile, not every external GPTQ SDK
option. Nonprojection weights stay floating. Llama's GGML Q/K permutation is
an explicit whole-row transform; codes/scales are never requantized.

`--maximum-statistics-bytes` and `--maximum-workspace-bytes` independently bound
each calibration matrix and provider-owned algorithm buffers (default 512 MiB
each). They exclude the model, caller-owned data, retained output payloads and
container publication; they are not process RSS limits. The safe CPU dense
factorization has cubic cost in input width and does not claim tuned BLAS/GPU
quantization performance.

Optional validation compares every prompt's last-token full vocabulary against
the floating source. Default gates are KL <= 0.5 nats and Top-10 overlap >= 6,
with at most 16 validation prompts. `--max-validation-prompts` changes that
explicit bound. Threshold flags require `--validation-dataset`. Cross-precision
generated tokens need not match. With no validation set the report states
`quality_status=not_evaluated`; artifact parity still runs on a calibration
prompt and is not presented as a quality measurement.

Select `--qif-profile single|package` and `--qif-codec none|zstd|lz4` as for the
other converter modes. PACKAGE additionally accepts `--max-shard-bytes`.
Before publication, the private candidate must pass full payload verification,
exact packed-byte checks and reload/full-logits/generation/KV parity with the
in-memory quantized model. Quality failure publishes no candidate. Existing
output/report paths are rejected. An optional JSON report is a separate
sidecar written after artifact publication; a sidecar failure does not roll
back a published verified artifact.

Reports record source inventory and decoded-Core projection hashes, exact
dataset hashes, options, calibration row counts, packed hashes, weight errors,
quality gates and artifact parity. An inventory hash identifies the recorded
controls/tensors; it is not the original container's whole-file hash.
Native QIF typed metadata and source/tokenizer revisions are retained.

Small tests cover both architectures, both dimension profiles, both bit widths,
all six profile/codec choices, native inputs, failed gates and no requantization.
The separately ignored `gptq_model_conformance` test quantizes all 210 SmolLM2
projections and checks four fixed validation inputs and two QIF profiles. The
[prompt provenance](../../tests/fixtures/gptq/README.md) states the development
sample boundary. Fixed-prompt acceptance does not establish general task accuracy.
