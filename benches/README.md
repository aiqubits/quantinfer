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
capacity. It also records the CPU model, source revision, and GGUF hash. Energy
is marked unmeasured.

Model Conformance publishes reports for the F32, Q8_0 and Q4_0 variants of
both models. SmolLM2 quantized runs use the GGML output-row layout.

The `baselines/` directory records the i7-12700 single-thread measurements
and the available GitHub runner measurements.

## llama.cpp throughput comparison

`llama-cpp-compare.py` runs QuantInfer and `llama-bench` from the llama.cpp
revision pinned by the GGUF Interoperability workflow
(`0ec4027a2a0183de87c007069adddc06e665f7de`) on the same GGUF, sweeping
thread counts. QuantInfer follows the real-model convention (five fixed
prompt tokens, eight greedy tokens, 64-token context); llama-bench uses the
same lengths with `-ngl 0`, so this is a CPU throughput comparison, not a
token-equivalence check (that is `tests/conformance/gguf_interop.py`).

```bash
python benches/llama-cpp-compare.py \
  --cli target/release/quantinfer-cli \
  --llama-bench /path/to/llama-bench \
  --model /path/to/model.gguf \
  --threads 1,2,4,8 \
  --output benches/llama-cpp/report.json
```

First recorded run, before output-column splitting (Apple M4 Pro,
SmolLM2-135M Q8_0 quantized with the pinned `llama-quantize`, QuantInfer
default scalar build; `llama-cpp/apple-m4-pro-smollm2-q8_0.json`):

| Threads | QuantInfer decode | llama.cpp decode | Ratio |
| --- | --- | --- | --- |
| 1 | 13.2 tok/s | 348.6 tok/s | 26.4x |
| 2 | 13.2 tok/s | 502.8 tok/s | 38.0x |
| 4 | 13.2 tok/s | 542.7 tok/s | 41.0x |
| 8 | 13.2 tok/s | 513.6 tok/s | 39.0x |

Decode did not scale because CPU matmul split input rows, and decode has
one row. Quantized matmul, including GGML output-row weights, now splits
output columns when there are fewer input rows than threads. The same
machine afterwards
(`llama-cpp/apple-m4-pro-smollm2-q8_0-column-split.json`):

| Threads | QuantInfer decode | llama.cpp decode | Ratio |
| --- | --- | --- | --- |
| 1 | 19.5 tok/s | 424.7 tok/s | 21.8x |
| 2 | 28.4 tok/s | 532.6 tok/s | 18.7x |
| 4 | 39.5 tok/s | 609.3 tok/s | 15.4x |

Four threads decode at about twice the one-thread rate. The single-thread
gap is still about 22x, so attention fusion remains the next lever. The
llama.cpp Comparison workflow reruns this comparison on the pinned
artifact. Thread counts above the performance-core count are not part of
the comparison: this machine mixes in efficiency cores past 10 threads.

Model Conformance checks peak RSS against a 10% allowance and verifies exact
KV cache sizes. On a CPU with a recorded timing baseline, it allows up to 35%
more prefill, first-token, and decode latency, and requires at least 70% of
baseline decode throughput. On other CPUs of the same architecture, it still
checks memory and publishes the timings.

| Model | Variant | Quality | Performance and memory | llama.cpp comparison |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | F32, Q8_0, Q4_0 | Transformers logits, prompt tokens, generation and EOS | Prefill, decode, peak RSS and KV cache | |
| Qwen3-0.6B | Q4_K | Weight and logit error, generation and EOS | | |
| SmolLM2-135M | F32, Q8_0, Q4_0 | Sampled logits, prompt tokens and generation | Prefill, decode, peak RSS and KV cache | GGML exports |
| SmolLM2-135M | External F16 | Sampled logits, prompt tokens and generation | | |
| SmolLM2-135M | External Q8_0, Q4_0 | | | Metadata, tensor layout and generated tokens |

The SmolLM2 quantized CLI tests also verify termination by setting the EOS ID
to the first expected token for the fixed prompt.
