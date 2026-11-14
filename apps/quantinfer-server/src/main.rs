//! Optional HTTP inference service.

mod config;
mod engine;
mod rate;
mod registry;
mod routes;
mod state;

use std::env;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tokio::net::TcpListener;

use config::Settings;
use engine::Metrics;
use rate::RateLimit;
use registry::ModelRegistry;
use routes::{error_response, generate, list_models, metrics_response, studio, studio_registry};
use state::AppState;
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
        .route("/studio/api/registry", get(studio_registry))
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

#[cfg(test)]
mod tests {
    use super::{config::Settings, routes::STUDIO_HTML};

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
        assert!(STUDIO_HTML.contains("/studio/api/registry"));
        assert!(STUDIO_HTML.contains("Live generation samples"));
        assert!(STUDIO_HTML.contains("tokens/s"));
    }
}
