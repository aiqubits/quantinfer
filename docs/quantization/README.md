# Quantization

## Collect calibration weights

Run the model on representative prompts to measure each input channel's
mean squared activation. Put one JSON string or one object with a
`text` string on each line of the dataset.

```sh
quantinfer collect-imatrix --input model-f32.gguf --dataset prompts.jsonl \
  --output imatrix.json --max-sequence-length 512
```

Collection supports Qwen3 and Llama, with an embedded tokenizer and
GGUF, QIF or a verified package as input. It starts a fresh KV cache
for each prompt and records attention and feed-forward projection inputs.
Each token contributes one observation. The default context limit is
512 tokens or the model limit if it is smaller. The collector rejects
oversized prompts and writes the output only after all samples succeed.

## Importance-weighted Q4_0

For a Llama or Qwen3 checkpoint exported with GGML output rows, supply
`--imatrix IMATRIX.gguf` with `--gguf-layout ggml --quantize q4_0`.
Use a GGUF calibration artifact from
[llama-imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix).

Pass the collector's output to the converter with `--imatrix imatrix.json`.
The JSON maps each quantized projection name, such as
`blk.0.attn_q.weight`, to an array of input-channel weights. Use the mean
square of each input activation over the calibration samples. Each array
must match the projection's input width and contain finite, nonnegative
weights with at least one positive entry.

For Qwen3, use `--tokenizer-pre qwen2` with the checkpoint's tokenizer.

```sh
quantinfer-convert --config config.json --weights model.safetensors \
  --tokenizer tokenizer.json --tokenizer-pre qwen2 --gguf-layout ggml \
  --quantize q4_0 --imatrix imatrix.json --output-format gguf --output model-q4_0.gguf
```

The converter evaluates several clipping points for each Q4_0 block and
keeps the scale with the smallest weighted reconstruction error. It
records the SHA-256 of the calibration input in the GGUF metadata. The
result uses the standard packed Q4_0 format.

## Automatic mixed-precision policy

Use a policy when attention projections need more precision than feed-forward
projections. The policy applies by canonical projection suffix and preserves
embeddings and the output head by default.

```json
{
  "default": "q4_k",
  "attention": "q8_0",
  "feed_forward": "q4_k",
  "preserve_output": true
}
```

```sh
quantinfer-convert --config config.json --weights model.safetensors \
  --quantize q4_k --mixed-precision-policy policy.json --output-format gguf --output model-mixed.gguf
```

The policy and explicit `--mixed-precision` plans are mutually exclusive.
The converter validates block alignment and records the resulting tensor
schemes in the model manifest.

## Native K-quant storage and quality checks

Q4_K, Q5_K and Q6_K select integer codes against the final stored codebook:
F16 global scales and the quantized subblock scale/min coefficients. Scales
outside finite F16 range are rejected. A positive global quotient that rounds
to zero uses the smallest positive F16 scale; smaller unresolvable weights may
still become zero. Independent scalar enumeration tests check nearest-code
selection, zero/subnormal inputs and nonrepresentable scales. They do not
establish that every model's task accuracy improves.

The fixed-model `k_quant_conformance` test extends SmolLM2-135M and Qwen3-0.6B
with native row-major Q4_K/Q5_K/Q6_K artifacts. The existing model-specific
Transformers/F32 tests and their thresholds remain in place. For the new fixed
prompt matrix, full-vocabulary next-token KL from F32 is limited to 0.5 nats
and Top-10 overlap is at least 6. SmolLM2's query projection relative L2 is at
most 0.25; Qwen3 additionally retains the existing Q4_K full-logit max/MSE
limits of 13/9 and query-weight max/mean limits of 0.032/0.0025 for all three
new variants. Reports retain raw logit errors and sampled Transformers errors.
SmolLM2's existing GGML-layout Q4_0 raw-logit limit of 3.8 is not reused as a
threshold for this different native K profile.

The matrix checks finite full logits, repeatable generation, streaming tokens,
KV lengths and lossless publication to SINGLE/none and PACKAGE/zstd. Every
tensor's shape, dtype and raw SHA-256 must match its GGUF source, as must
decoded query values, logits and generated tokens. This is a bounded fixed
prompt regression matrix, not a perplexity benchmark or general task-quality
guarantee. Higher bit counts need not give monotonically better generation.

After downloading and verifying the pinned model and independently validating
its `{model}-f32.gguf` baseline, prepare and run the matrix:

```sh
python3 tests/conformance/k_quant_artifacts.py --model smollm2 \
  --source-dir /path/to/checkpoint --output-dir /path/to/artifacts \
  --converter target/release/quantinfer-convert --gguf-only
QUANTINFER_K_QUANT_MODEL=smollm2 \
QUANTINFER_K_QUANT_DIR=/path/to/artifacts \
QUANTINFER_K_QUANT_CONVERTER="$PWD/target/release/quantinfer-convert" \
QUANTINFER_K_QUANT_REPORT=/path/to/quality.json \
  cargo test --release --locked -p quantinfer-runtime \
    --test k_quant_conformance -- --ignored --nocapture
```

Use `--model qwen3` and `QUANTINFER_K_QUANT_MODEL=qwen3` for Qwen3. The script
verifies source files against the repository manifests and records source,
converter and artifact hashes. The test publishes QIF into a directory it
creates, checks and hashes both profiles, then releases that variant's copies
before moving on. It cleans its own directory on failure as well. To inspect
persistent QIF artifacts, omit `--gguf-only`, set `QUANTINFER_K_QUANT_QIF=1`,
and omit `QUANTINFER_K_QUANT_CONVERTER` when running the test.

`QuantizationProvider::error_report` measures the supplied in-memory data.
When evaluating exports, use the data re-read from the published artifact;
an F32 in-memory scale can change when stored as F16. A correct in-memory
report alone does not establish the exported artifact's reconstruction error.
`QuantizedData::with_storage_precision()` preserves codes and returns a copy
with the current block formats' F16 scales; Float remains F32.
`error_report_for_storage()` evaluates this precision without changing the
source Core tensor or existing quantizer output bytes. It detects underflow
and rejects unrepresentable F16 scales. Neither method verifies a file.

## Measured precision optimization

Use `--mixed-precision-optimize` to choose encodings from measured reconstruction
errors under a global byte budget. This mode is separate from suffix-based
policies and explicit assignments. It accepts floating SafeTensors inputs in
the native QuantInfer layout. Embeddings, the output head and other nonprojection
weights remain F32 and consume the fixed part of the budget.

```json
{
  "maximum_encoded_weight_bytes": 195000000,
  "candidates": ["q4_k", "q5_k", "q6_k", "q8_0"],
  "importance_weights": "optimized-imatrix.json",
  "max_history_states": 4194304
}
```

The optional importance path is relative to the policy file and uses the
collector's JSON or the supported GGUF imatrix input. It must cover every
projection exactly. Without importance weights the objective is ordinary
reconstruction SSE. With them, it is the diagonal activation proxy
`sum_i E[x_i²] * sum_j (W[i,j] - restored[i,j])²` for native `[input, output]`
weights. It does not model channel correlations or directly optimize task
accuracy. Candidate evaluation includes F16 storage rounding and records
unaligned/nonrepresentable candidates as unavailable. F32 is eligible only
when explicitly listed as `f32`.

```sh
quantinfer-convert --config config.json --weights model.safetensors \
  --tokenizer tokenizer.json --mixed-precision-optimize policy.json \
  --mixed-precision-report NEW_REPORT.json \
  --output-format gguf --output model-optimized.gguf
```

QIF publication uses the same options with `--output-format qif` and its normal
profile/codec options. The report must be a new file, distinct from the artifact,
manifest and checksum output. Other quantization/mixed/imatrix flags cannot be
combined with optimization; the policy owns those decisions.

The budget is the sum of encoded **logical tensor payload** bytes, including
fixed F32 tensors. It excludes alias deduplication, container metadata/alignment,
codec effects, decoded Core storage, KV and workspace. It is neither a file-size
nor a process-memory limit. This makes the same plan usable across containers;
QIF deduplication may reduce the physical payload further.

The solver keeps a Pareto frontier and returns the minimum summed measured F64
objective among the supplied candidates that fit this budget. It prefers fewer
bytes on equal error and then deterministic traversal order. It does not use a
greedy upgrade heuristic or silently fall back to an approximate plan. The
default frontier/history limits are 65,536/1,048,576 states; configurable ceilings
are 1,048,576/4,194,304. Infeasible budgets, exhausted limits and unrepresentable
aggregate objectives fail explicitly. Independent exhaustive assignments test
the optimum, including a case where greedy upgrades lose.

Planning reads projections before the writing pass and rechecks their F32 input
hashes during writing. Every candidate is quantized from floating weights;
the chosen export is not quantized from another quantized candidate. Reports
record source projection hashes, per-candidate costs/errors, unavailable choices,
selections, limits, policy/calibration hashes and the published artifact hashes.
This adds conversion work and does not imply faster model loading or inference.

The fixed SmolLM2 CI case calibrates on three declared prompts, selects among
four encodings under 195,000,000 logical payload bytes, and checks the written
artifact against F32, SINGLE/none and PACKAGE/zstd. It retains full logits,
KL/Top-K, weight error, generation and KV checks. This is a fixed-model
acceptance case, not a universal quality guarantee for other policies/models.
