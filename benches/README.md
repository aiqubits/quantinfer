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

Model Conformance publishes separate reports for SmolLM2 F32 and Qwen3 F32,
Q8_0, and Q4_0. Compare performance runs on the same CPU model.
