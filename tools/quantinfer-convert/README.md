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

See [AWQ](#awq) for activation-aware scaling and clipping.

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
The separately ignored `model_quantization_conformance` test quantizes all 210 SmolLM2
projections and checks four fixed validation inputs and two QIF profiles. The
[prompt provenance](../../tests/fixtures/gptq/README.md) states the development
sample boundary. Fixed-prompt acceptance does not establish general task accuracy.

## AWQ

Build with `--features awq` and select `quantinfer-convert awq` with the
same floating input, JSONL calibration and validation, and QIF publication
options described above. The signed native profile supports `--bits 4|8`,
`--group-size` in multiples of 32, and `--grid-points 2..100` (default 20).
Clipping is enabled by default. Use `--no-clipping` to measure scaling alone.

Calibration propagates the actual packed layer outputs in sequence. It
searches Q/K/V scales using the complete attention output and gate/up
scales using the complete SwiGLU output. The down projection uses a linear
objective and folds compensation into the up projection. V/O compensation
uses a one-to-one scale mapping and is skipped when grouped-query attention
repeats value heads. Clipping uses every supplied token row, selects bounds
against within-group reconstruction loss, and leaves Q/K out of the search.
All these steps use floating candidates. Final packing writes the chosen
signed grids directly to Q4_0 or Q8_0.

AWQ publishes the two folded norms per layer as F32, including when their
source storage is F16 or BF16. The report includes source and transformed
projection hashes, selected ratios and losses, clipping losses, and the
replacement normalization hashes. Other nonprojection tensors retain their
payloads. `--maximum-statistics-bytes` bounds the activation workspace
estimate and `--maximum-workspace-bytes` bounds the provider's search buffers.
These bounds exclude model storage, retained payloads, optional quality
validation and publication.

The [AWQ adapter](../../adapters/awq/README.md) documents the exact grid,
upstream algorithm references and independent Torch vectors. The full
SmolLM2 test uses the same 13 calibration and four fixed validation inputs
as the GPTQ test.
