//! Optional HTTP inference service.

mod engine;
mod rate;
mod registry;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::env;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use quantinfer::runtime::{DecodeEvent, DecodeSession, DecodeStopReason, GenerationConfig};
use quantinfer::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use engine::{Metrics, Submission};
use rate::RateLimit;
use registry::ModelRegistry;

struct AppState {
    models: Arc<ModelRegistry>,
    queue: mpsc::Sender<Submission>,
    metrics: Arc<Metrics>,
    context_length: usize,
    request_timeout: Duration,
    rate_limit: Option<Mutex<RateLimit>>,
}

struct Settings {
    bind: String,
    models: BTreeMap<String, PathBuf>,
    model_registry: Option<PathBuf>,
    context_length: usize,
    maximum_requests: usize,
    batch: engine::BatchLimits,
    kv_budget_bytes: usize,
    request_timeout: Duration,
    requests_per_second: Option<NonZeroUsize>,
}

impl Settings {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut settings = Self {
            bind: "127.0.0.1:8080".to_owned(),
            models: BTreeMap::new(),
            model_registry: None,
            context_length: 512,
            maximum_requests: 8,
            batch: engine::BatchLimits {
                sequences: NonZeroUsize::new(8).unwrap(),
                tokens: NonZeroUsize::new(512).unwrap(),
            },
            kv_budget_bytes: 512 * 1024 * 1024,
            request_timeout: Duration::from_secs(60),
            requests_per_second: None,
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            let number = || {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {flag}: {error}"))
            };
            match flag.as_str() {
                "--bind" => settings.bind = value,
                "--context-length" => settings.context_length = number()?,
                "--max-requests" => settings.maximum_requests = number()?,
                "--max-batch-sequences" | "--max-batch-tokens" => {
                    let limit = NonZeroUsize::new(number()?)
                        .ok_or_else(|| format!("{flag} must be positive"))?;
                    if flag == "--max-batch-sequences" {
                        settings.batch.sequences = limit;
                    } else {
                        settings.batch.tokens = limit;
                    }
                }
                "--kv-budget-bytes" => settings.kv_budget_bytes = number()?,
                "--requests-per-second" => {
                    settings.requests_per_second = Some(
                        NonZeroUsize::new(number()?)
                            .ok_or_else(|| format!("{flag} must be positive"))?,
                    );
                }
                "--request-timeout-ms" => {
                    settings.request_timeout = Duration::from_millis(number()? as u64)
                }
                "--model-registry" => settings.model_registry = Some(PathBuf::from(value)),
                "--model" => {
                    let (name, path) = value
                        .split_once('=')
                        .filter(|(name, path)| !name.is_empty() && !path.is_empty())
                        .ok_or_else(|| "--model requires NAME=PATH".to_owned())?;
                    if settings
                        .models
                        .insert(name.to_owned(), PathBuf::from(path))
                        .is_some()
                    {
                        return Err(format!("duplicate model name {name}"));
                    }
                }
                _ => return Err(format!("unknown server option {flag}")),
            }
        }
        if settings.context_length == 0
            || !(1..=1024).contains(&settings.maximum_requests)
            || settings.kv_budget_bytes == 0
            || settings.request_timeout.is_zero()
        {
            return Err("context, request timeout and KV budget must be positive and --max-requests must be 1..=1024".to_owned());
        }
        Ok(settings)
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("quantinfer-server [--bind ADDRESS] [--model NAME=PATH] [--model-registry FILE] [--context-length N] [--max-requests N] [--max-batch-sequences N] [--max-batch-tokens N] [--kv-budget-bytes N] [--request-timeout-ms N] [--requests-per-second N]");
        return;
    }
    if let Err(error) = run(args).await {
        eprintln!("quantinfer-server: {error}");
        std::process::exit(1);
    }
}

async fn run(args: Vec<String>) -> Result<(), String> {
    let settings = Settings::parse(args)?;
    let models = Arc::new(ModelRegistry::new(
        settings.models,
        settings.model_registry,
    )?);
    #[cfg(unix)]
    let reload = models.watch()?;
    let listener = TcpListener::bind(&settings.bind)
        .await
        .map_err(|error| format!("cannot bind {}: {error}", settings.bind))?;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    let metrics = Arc::new(Metrics::default());
    let (queue, worker) = engine::start(
        settings.maximum_requests,
        settings.kv_budget_bytes,
        settings.batch,
        metrics.clone(),
    )?;
    let state = Arc::new(AppState {
        models,
        queue,
        metrics,
        context_length: settings.context_length,
        request_timeout: settings.request_timeout,
        rate_limit: settings
            .requests_per_second
            .map(|limit| Mutex::new(RateLimit::new(limit, Instant::now()))),
    });
    let router = Router::new()
        .route("/healthz", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(list_models))
        .route("/v1/generate", post(generate))
        .route("/metrics", get(metrics_response))
        .route("/studio", get(studio))
        .fallback(|| async { error_response((StatusCode::NOT_FOUND, "not_found".to_owned())) })
        .layer(DefaultBodyLimit::max(64 * 1024))
        .with_state(state);
    println!("quantinfer-server listening on {address}");
    let served = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| error.to_string());
    #[cfg(unix)]
    if let Some(reload) = reload {
        reload.abort();
    }
    served?;
    worker
        .join()
        .map_err(|_| "decode worker panicked".to_owned())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    if let Ok(mut terminate) =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
    {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
        return;
    }
    let _ = tokio::signal::ctrl_c().await;
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let data: Vec<_> = state
        .models
        .names()
        .into_iter()
        .map(|id| json!({"id": id, "object": "model"}))
        .collect();
    Json(json!({"data": data}))
}

async fn metrics_response(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let metrics = &state.metrics;
    let body = format!(
        "quantinfer_requests_total {}\nquantinfer_requests_rejected_total {}\nquantinfer_requests_finished_total {}\nquantinfer_generated_tokens_total {}\nquantinfer_batched_forward_steps_total {}\n",
        metrics.requests.load(Ordering::Relaxed),
        metrics.rejected.load(Ordering::Relaxed),
        metrics.finished.load(Ordering::Relaxed),
        metrics.generated_tokens.load(Ordering::Relaxed),
        metrics.batched_steps.load(Ordering::Relaxed),
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}

/// Serve a dependency-free local dashboard for model registry and inference
/// smoke tests. The page deliberately uses the public HTTP endpoints so it is
/// also useful as a small integration example for clients.
async fn studio() -> Html<&'static str> {
    Html(STUDIO_HTML)
}

const STUDIO_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantInfer Studio</title>
  <style>
    :root { color-scheme: light dark; font: 16px/1.45 system-ui, sans-serif; }
    body { margin: 0 auto; max-width: 1050px; padding: 2rem; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
    h1 { margin: 0; }
    .status { color: #087f5b; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    section { border: 1px solid #8886; border-radius: .7rem; padding: 1rem; }
    label { display: block; margin: .6rem 0 .2rem; }
    textarea, input, select, button { box-sizing: border-box; font: inherit; width: 100%; padding: .5rem; }
    textarea { min-height: 8rem; resize: vertical; }
    button { cursor: pointer; margin-top: .8rem; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #8882; padding: .7rem; border-radius: .4rem; }
    code { font-size: .9em; }
  </style>
</head>
<body>
  <header>
    <h1>QuantInfer Studio</h1>
    <span id="health" class="status">checking service…</span>
  </header>
  <p>Inspect the live model registry, run a generation smoke test, and view
     runtime counters without installing a separate frontend.</p>
  <div class="grid">
    <section>
      <h2>Model registry</h2>
      <p id="models">loading…</p>
      <button id="refresh" type="button">Refresh</button>
    </section>
    <section>
      <h2>Generation</h2>
      <label for="model">Model</label>
      <select id="model"></select>
      <label for="prompt">Prompt</label>
      <textarea id="prompt" placeholder="Write a prompt or use token IDs below"></textarea>
      <label for="max-tokens">Max new tokens</label>
      <input id="max-tokens" type="number" min="1" max="4096" value="32">
      <button id="generate" type="button">Generate</button>
      <pre id="result">No request yet.</pre>
    </section>
    <section>
      <h2>Benchmark counters</h2>
      <pre id="metrics">loading…</pre>
      <p>For reproducible latency and memory measurements, use the pinned
        <code>quantinfer-xtask bench</code> command.</p>
    </section>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    async function refresh() {
      const [health, models, metrics] = await Promise.all([
        fetch('/healthz'), fetch('/v1/models'), fetch('/metrics')
      ]);
      $('health').textContent = health.ok ? 'service healthy' : 'service unavailable';
      const payload = await models.json();
      const entries = payload.data || [];
      $('models').textContent = entries.length ? entries.map((m) => m.id).join(', ') : 'No models loaded';
      $('model').replaceChildren(...entries.map((m) => new Option(m.id, m.id)));
      $('metrics').textContent = await metrics.text();
    }
    $('refresh').addEventListener('click', () => refresh().catch((e) => $('health').textContent = e));
    $('generate').addEventListener('click', async () => {
      const model = $('model').value;
      if (!model) { $('result').textContent = 'Load a model before generating.'; return; }
      $('result').textContent = 'running…';
      const response = await fetch('/v1/generate', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({model, prompt: $('prompt').value,
          max_new_tokens: Number($('max-tokens').value) || 32})
      });
      const body = await response.json();
      $('result').textContent = response.ok ? (body.text || JSON.stringify(body, null, 2))
        : (body.error || JSON.stringify(body, null, 2));
    });
    refresh().catch((e) => $('health').textContent = e);
  </script>
</body>
</html>"##;

#[derive(Deserialize)]
struct GenerateRequest {
    model: String,
    prompt: Option<String>,
    tokens: Option<Vec<usize>>,
    max_new_tokens: Option<usize>,
    eos_token_id: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

type ApiError = (StatusCode, String);

fn error_response((status, message): ApiError) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Response {
    state.metrics.requests.fetch_add(1, Ordering::Relaxed);
    if state.rate_limit.as_ref().is_some_and(|limit| {
        !limit
            .lock()
            .is_ok_and(|mut bucket| bucket.allow(Instant::now()))
    }) {
        state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "1")],
            Json(json!({"error": "request rate limit exceeded"})),
        )
            .into_response();
    }
    match start_request(&state, request).await {
        Ok(response) => response,
        Err(error) => {
            state.metrics.rejected.fetch_add(1, Ordering::Relaxed);
            error_response(error)
        }
    }
}

async fn start_request(state: &AppState, request: GenerateRequest) -> Result<Response, ApiError> {
    let bad = |message| (StatusCode::BAD_REQUEST, message);
    let model = state.models.get(&request.model).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("unknown model {}", request.model),
        )
    })?;
    let context = state
        .context_length
        .min(model.decoder.max_position_embeddings());
    let max_new_tokens = request.max_new_tokens.unwrap_or(32);
    if !(1..=4096).contains(&max_new_tokens) {
        return Err(bad("max_new_tokens must be 1..=4096".to_owned()));
    }
    let tokens = match (request.prompt, request.tokens) {
        (Some(prompt), None) => model
            .tokenizer
            .as_ref()
            .ok_or_else(|| bad("this model requires token IDs".to_owned()))?
            .encode(&prompt)
            .map_err(|error| bad(error.to_string()))?,
        (None, Some(tokens)) => tokens,
        _ => return Err(bad("provide exactly one of prompt or tokens".to_owned())),
    };
    let eos_token_id = request.eos_token_id.or_else(|| {
        model
            .tokenizer
            .as_ref()
            .and_then(|tokenizer| tokenizer.eos_token_id())
    });
    if tokens
        .iter()
        .chain(eos_token_id.iter())
        .any(|&token| token >= model.decoder.vocab_size())
    {
        return Err(bad("token ID is outside the model vocabulary".to_owned()));
    }
    let cache = model
        .decoder
        .new_cache(context)
        .map_err(|error| bad(error.to_string()))?;
    let session = DecodeSession::new(
        model.decoder.clone(),
        cache,
        tokens,
        GenerationConfig {
            max_new_tokens,
            eos_token_id,
            temperature: request.temperature.unwrap_or(0.0),
            top_k: request.top_k,
            top_p: request.top_p,
            seed: request.seed.unwrap_or(0),
        },
    )
    .map_err(|error| bad(error.to_string()))?;
    let (accepted, acceptance) = oneshot::channel();
    let (output, mut events) = mpsc::channel(max_new_tokens + 1);
    let deadline = Instant::now()
        .checked_add(state.request_timeout)
        .ok_or_else(|| bad("request deadline exceeds the clock range".to_owned()))?;
    state
        .queue
        .try_send(Submission {
            session,
            deadline,
            accepted,
            output,
        })
        .map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => (
                StatusCode::TOO_MANY_REQUESTS,
                "request queue is full".to_owned(),
            ),
            mpsc::error::TrySendError::Closed(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "decode worker is unavailable".to_owned(),
            ),
        })?;
    let deadline = tokio::time::Instant::from_std(deadline);
    let request_id = tokio::time::timeout_at(deadline, acceptance)
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "request timed out in the queue".to_owned(),
            )
        })?
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "decode worker is unavailable".to_owned(),
            )
        })?
        .map_err(|error| (StatusCode::TOO_MANY_REQUESTS, error))?;
    if request.stream {
        let stream = ReceiverStream::new(events).map(|event| Ok::<_, Infallible>(sse_event(event)));
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let mut tokens = Vec::new();
    let mut stopped = None;
    loop {
        let event = match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => break,
            Err(_) => {
                return Ok(error_response((
                    StatusCode::GATEWAY_TIMEOUT,
                    "generation timed out".to_owned(),
                )))
            }
        };
        match event {
            DecodeEvent::Token {
                token_id,
                stop_reason,
                ..
            } => {
                tokens.push(token_id);
                stopped = stop_reason;
            }
            DecodeEvent::Stopped { reason, .. } => stopped = Some(reason),
            DecodeEvent::Failed { error, .. } => {
                return Ok(error_response((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    error.to_string(),
                )));
            }
        }
        if stopped.is_some() {
            break;
        }
    }
    let Some(reason) = stopped else {
        return Ok(error_response((
            StatusCode::SERVICE_UNAVAILABLE,
            "decode worker stopped before completion".to_owned(),
        )));
    };
    if reason == DecodeStopReason::TimedOut {
        return Ok(error_response((
            StatusCode::GATEWAY_TIMEOUT,
            "generation timed out".to_owned(),
        )));
    }
    let text = model
        .tokenizer
        .as_ref()
        .map(|tokenizer| tokenizer.decode(&tokens))
        .transpose()
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(Json(json!({
        "request_id": request_id, "model": request.model, "tokens": tokens,
        "text": text, "finish_reason": stop_name(reason)
    }))
    .into_response())
}

fn stop_name(reason: DecodeStopReason) -> &'static str {
    match reason {
        DecodeStopReason::Eos => "eos",
        DecodeStopReason::MaxTokens => "length",
        DecodeStopReason::Cancelled => "cancelled",
        DecodeStopReason::TimedOut => "timeout",
        DecodeStopReason::Failed => "error",
    }
}

fn sse_event(event: DecodeEvent) -> Event {
    let (kind, data) = match event {
        DecodeEvent::Token {
            request_id,
            token_id,
            stop_reason,
        } => (
            "token",
            json!({"request_id": request_id, "token_id": token_id, "finish_reason": stop_reason.map(stop_name)}),
        ),
        DecodeEvent::Stopped { request_id, reason } => (
            "done",
            json!({"request_id": request_id, "finish_reason": stop_name(reason)}),
        ),
        DecodeEvent::Failed { request_id, error } => (
            "error",
            json!({"request_id": request_id, "error": error.to_string()}),
        ),
    };
    Event::default().event(kind).data(data.to_string())
}

#[cfg(test)]
mod tests {
    use super::{Settings, STUDIO_HTML};

    #[test]
    fn parses_multiple_models_and_rejects_invalid_limits() {
        let args = [
            "--model",
            "smol=smol.gguf",
            "--model",
            "qwen=qwen",
            "--max-requests",
            "4",
        ];
        let settings = Settings::parse(args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(settings.models.len(), 2);
        assert_eq!(settings.maximum_requests, 4);
        assert!(Settings::parse(["--max-requests", "0"].into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn studio_page_uses_public_runtime_endpoints() {
        assert!(STUDIO_HTML.contains("/v1/models"));
        assert!(STUDIO_HTML.contains("/v1/generate"));
        assert!(STUDIO_HTML.contains("/metrics"));
    }
}
