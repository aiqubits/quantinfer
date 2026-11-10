# Applications

User-facing executables such as the CLI and the optional HTTP server.

## HTTP inference

Start the server with one or more `--model NAME=PATH` options. Each path can
point to a GGUF file or a verified model package.

`--model-registry registry.json` also loads models from a JSON object keyed by
model name. Each entry contains `path`, an optional `revision`, and an optional
`lora` directory for a PEFT adapter trained with that base model. Paths in the
file resolve relative to the registry. On Unix, send `SIGHUP` after editing
entries. Change `revision` when replacing files at existing paths. The server
switches to the new registry after all changed models and adapters load
successfully. Active requests keep using the model they acquired at admission.

```json
{"smol": {"path": "models/smol.gguf", "revision": "1", "lora": "adapters/smol"}}
```

```sh
cargo run --release --locked -p quantinfer-server -- \
  --bind 127.0.0.1:8080 --model smol=/path/to/model.gguf --context-length 64
```

`POST /v1/generate` accepts `model` and exactly one of `prompt` or `tokens`.
It returns token IDs, decoded text when a tokenizer is available, and the
finish reason. Set `stream` to `true` to receive Server-Sent Events. Each
`token` event carries a token ID. The last token carries its finish reason,
and a timeout or decode failure emits a terminal event.

`GET /v1/models` lists loaded model names. `GET /healthz` reports service
health and `GET /metrics` exposes counters. Configure queue admission with
`--max-requests` and `--kv-budget-bytes`, and request deadlines with
`--request-timeout-ms`. `SIGINT` and `SIGTERM` stop accepting connections
and allow active responses to finish.

Set `--requests-per-second` to limit generation admissions across clients.
The bucket holds one second of burst capacity. Throttled requests return
HTTP 429 with a `Retry-After` header.

The decode worker batches ready requests for the same loaded model, including
new prompts and continuing decode steps. Each request keeps its own KV cache,
position and deadline. `--max-batch-sequences` and `--max-batch-tokens` default
to eight sequences and 512 input tokens. Larger prompts get an exclusive
forward step. `quantinfer_batched_forward_steps_total` counts grouped steps.
