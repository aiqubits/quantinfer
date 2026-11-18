# Benchmarks

Operator, runtime, quantization, and end-to-end performance measurements.

The first reproducible smoke benchmark covers Q8_0 quantization and CPU scalar
matrix multiplication.

```bash
cargo run -p quantinfer-xtask -- bench
```

It prints the iteration count, matrix shape, elapsed nanoseconds, and a numeric
checksum. The command is also executed by the GitHub Actions workflow.

## Real-model benchmark

Build the CLI in release mode and use a verified GGUF converted from the model
revision in `model-zoo/manifests`.

```bash
cargo build --release --locked -p quantinfer-cli
bash benches/real-model.sh target/release/quantinfer-cli \
  HuggingFaceTB/SmolLM2-135M 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 f32 /path/to/model.gguf
```

The benchmark uses five fixed input tokens, eight greedy output tokens, a
64-token context, and one CPU thread. It records model load time, prefill time,
time from prefill to the first selected token, throughput between the first
and last selected tokens, peak process RSS on Linux, and KV cache usage and
capacity. It also records the CPU model, source revision, GGUF hash, executable
hash and declared code commit (`QUANTINFER_BENCH_CODE_COMMIT`, or `GITHUB_SHA`
inside CI; otherwise `unknown`). Energy is marked unmeasured.

Model Conformance publishes reports for the F32, Q8_0 and Q4_0 variants of
both models. SmolLM2 quantized runs use the GGML output-row layout.

The `baselines/` directory records the i7-12700 single-thread measurements
and the available GitHub runner measurements.

Model Conformance checks peak RSS against a 10% allowance and verifies exact
KV cache sizes. On a CPU with a recorded timing baseline, it allows up to 35%
more prefill, first-token, and decode latency, and requires at least 70% of
baseline decode throughput. On other CPUs of the same architecture, it still
checks memory and publishes the timings.

To avoid skipping timing gates when GitHub assigns an unrecorded CPU, Model
Conformance also rebuilds pinned commit
`3cd5745116c3e5e6edf3ec20b943ddec201b4686` in a separate target directory with
the same Rust toolchain. For each model's F32, Q8_0 and Q4_0 variants, it runs
three fresh processes per executable, interleaving reference/current order,
and gates per-metric medians using the same limits above. Both sides use the
exact same GGUF file, CPU, token sequence and runtime configuration. Reports
must agree on these identities, KV sizes and generated tokens; missing or
non-finite metrics fail. Raw samples, medians, code commits and binary hashes
are uploaded. This measures regression relative to that fixed implementation;
it is not an absolute performance guarantee. File caches and host load remain
uncontrolled, and historical baseline checks still run separately.

```sh
python3 benches/paired-model.py --current-cli /path/current/quantinfer-cli \
  --current-commit FULL_SHA --reference-cli /path/reference/quantinfer-cli \
  --reference-commit 3cd5745116c3e5e6edf3ec20b943ddec201b4686 \
  --model-id HuggingFaceTB/SmolLM2-135M \
  --source-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --variant f32 --model /path/model.gguf --output-prefix /tmp/smollm2-paired
```

| Model | Variant | Quality | Performance and memory | llama.cpp comparison |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | F32, Q8_0, Q4_0 | Transformers logits, prompt tokens, generation and EOS | Prefill, decode, peak RSS and KV cache | |
| Qwen3-0.6B | Q4_K | Weight and logit error, generation and EOS | | |
| Qwen3-0.6B | Native Q4_K, Q5_K, Q6_K | Fixed-prompt full logits/KL/Top-K, query-weight error, streaming/KV and QIF parity | | |
| SmolLM2-135M | F32, Q8_0, Q4_0 | Sampled logits, prompt tokens and generation | Prefill, decode, peak RSS and KV cache | GGML exports |
| SmolLM2-135M | Native Q4_K, Q5_K, Q6_K | Fixed-prompt full logits/KL/Top-K, query-weight error, streaming/KV and QIF parity | | |
| SmolLM2-135M | External F16 | Sampled logits, prompt tokens and generation | | |
| SmolLM2-135M | External Q8_0, Q4_0 | | | Metadata, tensor layout and generated tokens |

The SmolLM2 quantized CLI tests also verify termination by setting the EOS ID
to the first expected token for the fixed prompt.

The native K matrix retains the original model-specific gates and adds the
budgets and artifact publication checks documented in
[quantization](../docs/quantization/README.md#native-k-quant-storage-and-quality-checks).
Its one-prompt probability comparison is not a task-accuracy/perplexity
benchmark. No K-format timing, RSS regression or llama.cpp parity gate is
implied by these quality rows.

## QIF / SafeTensors / GGUF storage measurements

```sh
cargo run --release --locked -p quantinfer-xtask -- qif-bench \
  --qif /path/to/model.qif --gguf /path/to/model.gguf \
  --safetensors /path/to/model.safetensors \
  --qif-tensor token_embd.weight --gguf-tensor token_embd.weight \
  --safetensors-tensor model.embed_tokens.weight
```

Each artifact runs in a fresh child process so prior format allocations and
allocator state do not contaminate its RSS/page-fault measurements. The output
records process IDs, selected names, dtypes, raw tensor sizes, range byte counts
and hashes. Select corresponding tensors/variants explicitly; format-default
ordering is not a semantic mapping. Layout or quantization changes can make
raw range hashes differ even for related model weights.
The pinned SmolLM2 source checkpoint and converted artifacts can also have
different storage precision. Its CI selects the corresponding embedding by
name and requires identical QIF/GGUF range bytes; the original SafeTensors
dtype and range hash are reported, not assumed equal to converted F32 bytes.

`open_ns` includes all file reads performed while opening the reader, including
the current GGUF reader's eager full-file read. QIF and SafeTensors use their
file-backed control readers. All first-range operations materialize at most
4096 bytes; SafeTensors range timing includes opening/seeking its payload file.
`open_to_first_range_ns` measures the full interval including tensor lookup.
The different reader strategies are printed and should be considered when
interpreting individual phase timings.

QIF full verification is measured separately from open/range, including its own
resource deltas. SafeTensors and GGUF report checksum verification unsupported.
On Linux, RSS/page-fault counters come from procfs; `peak_rss_kb` is the process
high-water mark at the end of a phase, not an independently reset phase peak.
RSS deltas may be negative. Other platforms may omit these counters.

The OS file cache is **uncontrolled**. Process isolation does not create cold
storage caches or remove system load. These measurements compare the supplied
artifacts and current reader implementations; they do not establish an
intrinsic format speed advantage. Without paths the command remains an
in-memory synthetic smoke measurement, labelled `synthetic_memory`.
