//! HTTP route composition and shared response helpers.

mod generate;
mod studio;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::state::AppState;

pub(super) type ApiError = (StatusCode, String);

pub(crate) use generate::generate;
#[cfg(test)]
pub(crate) use studio::STUDIO_HTML;
pub(crate) use studio::{studio, studio_registry};

pub(crate) async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let data: Vec<_> = state
        .models
        .names()
        .into_iter()
        .map(|id| json!({"id": id, "object": "model"}))
        .collect();
    Json(json!({"data": data}))
}

pub(crate) async fn metrics_response(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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

pub(crate) fn error_response((status, message): ApiError) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}
