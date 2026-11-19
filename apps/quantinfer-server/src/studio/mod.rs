//! Local Studio management endpoints.

mod jobs;
mod models;
mod profiles;

use std::sync::Arc;

use crate::registry::ModelRegistry;
use axum::extract::{Path, State};
use axum::http::{header::HOST, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

pub(crate) use jobs::Jobs;

pub(crate) fn router(jobs: Arc<Jobs>, registry: Arc<ModelRegistry>) -> Router {
    let models = models::router(jobs.clone(), registry);
    let profiles = profiles::router(jobs.clone());
    Router::new()
        .route("/studio/api/jobs", get(list).post(submit))
        .route("/studio/api/jobs/{id}", delete(cancel))
        .route("/studio/api/inventory", get(inventory))
        .route("/studio/api/datasets/{name}", put(save_dataset))
        .with_state(jobs)
        .merge(models)
        .merge(profiles)
}

async fn inventory(State(jobs): State<Arc<Jobs>>, headers: HeaderMap) -> Response {
    if !authorized(&jobs, &headers, false) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match jobs.inventory() {
        Ok(inventory) => Json(inventory).into_response(),
        Err(error) => failure(StatusCode::INTERNAL_SERVER_ERROR, &error),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    text: String,
}

async fn save_dataset(
    State(jobs): State<Arc<Jobs>>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(dataset): Json<Dataset>,
) -> Response {
    if !authorized(&jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match jobs.save_dataset(&name, &dataset.text) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => failure(StatusCode::BAD_REQUEST, &error),
    }
}

async fn list(State(jobs): State<Arc<Jobs>>, headers: HeaderMap) -> Response {
    if !authorized(&jobs, &headers, false) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    Json(json!({"jobs": jobs.list()})).into_response()
}

async fn submit(
    State(jobs): State<Arc<Jobs>>,
    headers: HeaderMap,
    Json(request): Json<jobs::Request>,
) -> Response {
    if !authorized(&jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match jobs.submit(request) {
        Ok(job) => (StatusCode::ACCEPTED, Json(job)).into_response(),
        Err(error) => failure(StatusCode::BAD_REQUEST, &error),
    }
}

async fn cancel(
    State(jobs): State<Arc<Jobs>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    if jobs.cancel(&id) {
        return StatusCode::ACCEPTED.into_response();
    }
    failure(StatusCode::CONFLICT, "job does not accept cancellation")
}

fn authorized(jobs: &Jobs, headers: &HeaderMap, change: bool) -> bool {
    headers
        .get(HOST)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|host| jobs.accepts_host(host))
        && (!change
            || headers
                .get("x-quantinfer-studio")
                .is_some_and(|value| value == "1"))
}

fn failure(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}
