# Deployment

QuantInfer can run as a local CLI, an HTTP service, a C ABI library, a WASM
module, or a WASI-NN graph. The core crates do not require a GPU, CANN, Python,
or a network connection at runtime.

## Model package

Prefer a verified model package for deployment:

```text
model/
├── manifest.json
├── model.gguf
├── tokenizer.json
├── tokenizer_config.json
├── checksums.sha256
└── profiles/
```

Create and verify a package with `quantinfer-convert` and
`quantinfer-cli verify-package`. The manifest binds the model architecture,
source revision, tensor descriptors, tokenizer revision, checksums, and
inference profiles. A plain GGUF file remains supported for compatibility.

## HTTP service

Start a single model:

```sh
cargo run --release --locked -p quantinfer-server -- \
  --bind 127.0.0.1:8080 \
  --model smol=/srv/models/smol.gguf \
  --context-length 2048
```

Use `--model NAME=PATH` more than once or provide a JSON registry with
`--model-registry`. Registry paths are resolved relative to the registry file;
on Unix, send `SIGHUP` after changing an entry. A revision change causes the
new model and optional LoRA adapter to load atomically while admitted requests
continue on their existing model.

The service exposes `GET /healthz`, `GET /v1/models`, `POST /v1/generate`, and
`GET /metrics`. Set `stream: true` on generation requests for Server-Sent
Events. `GET /studio` serves a dependency-free local dashboard for registry,
generation smoke tests, and metrics. Configure queue limits, KV memory budget,
request deadlines, and rate limiting with the corresponding server flags.

The service handles SIGINT/SIGTERM gracefully. Keep it behind the deployment's
authentication and TLS layer when exposed beyond localhost; the built-in
server intentionally focuses on inference scheduling rather than identity.

## Native targets

Build only the target you deploy. CUDA, Metal, WGPU, and Ascend providers are
optional and dynamically load their vendor runtime where required. WASM and
WASI-NN are independent workspace targets; C and Python bindings call the
stable C ABI and do not pull optional GPU dependencies into the core library.
