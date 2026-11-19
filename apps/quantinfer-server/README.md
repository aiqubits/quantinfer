# QuantInfer Studio

Start the Studio management tools on a loopback listener. The configured CLI
and converter must be built from a version that provides the commands below.

```sh
cargo build --release --locked -p quantinfer-server -p quantinfer-cli
cargo build --release --locked -p quantinfer-convert --features gptq,awq,autoround
target/release/quantinfer-server --bind 127.0.0.1:8080 \
  --studio-dir "$PWD/studio-data" \
  --studio-cli "$PWD/target/release/quantinfer-cli" \
  --studio-converter "$PWD/target/release/quantinfer-convert"
```

Open `/studio` on that address. Copy a model artifact into `studio-data/sources`
and PEFT adapter directories into `studio-data/adapters`. Keep the Studio data
directory under the operator's control. Import prompts from JSONL files in
`datasets` or save them from the page. Each line can be a string or an object
with a `text` string.

## Workflows

- Run GPTQ, AWQ or AutoRound on floating Dense weights with a calibration set
  and an optional validation set. The converter verifies the published QIF
  weights and reload parity. Collect imatrix for other quantization workflows.
- Create QIF packages from GGUF, re-shard native QIF with any core codec, and
  check native QIF or legacy model packages. Native repacking verifies the
  input first and produces an unsigned package with its own checksums.
- Register a local or produced model, optionally with a PEFT adapter and a
  stored profile. Studio loads and validates the replacement before updating
  the registry. Existing requests retain their active model. Removing a
  registration leaves the weight artifact available for reuse.
- Save an inference profile with context, CPU threads and an optional memory
  budget. It records the model artifact's SHA-256 identity and binds a CPU
  execution variant. For native QIF with no declared variant, Studio resolves
  a local CPU variant against the embedded manifest before accepting it.
- Benchmark a specific model with a fixed prompt, token limit and context. The
  tool records the request and separates load, prefill, first-token and decode
  measurements. The report includes decode throughput, KV usage, peak RSS
  when available, input identity and the CLI executable hash. Cache state is
  uncontrolled.

Jobs have queued, running, succeeded, failed or cancelled states. One offline
job runs at a time. You can cancel queued or running jobs. Failed or cancelled
jobs discard their staging directory. Complete artifacts and reports live in
`studio-data/artifacts/ID`, and successful records are available after restart.
The web form can use a published QIF directly as the input to another job or
to a managed registration. Inference and the status dashboard also work
without enabling the local management routes.

The same package and benchmark operations are available from the shell.

```sh
target/release/quantinfer-convert repack-qif --input model.qif \
  --output package.qif --qif-codec zstd --max-shard-bytes 67108864
target/release/quantinfer-cli generate --input package.qif --prompt 'Write a short answer.' \
  --max-new-tokens 4 --max-sequence-length 32 --profile-report benchmark.json
```

The fixed SmolLM2 integration test exercises real calibration and quantization,
cancel and failure cleanup, repacking, profile and LoRA serving, benchmark
reports, and persistence across a fresh server process.
