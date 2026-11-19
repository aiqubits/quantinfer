//! Persist locally managed model registrations after loading the replacement.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::put;
use axum::{Json, Router};
use serde::Deserialize;

use super::{authorized, failure, jobs::validate_name, profiles, Jobs};
use crate::registry::{ModelEntry, ModelRegistry};

struct Management {
    jobs: Arc<Jobs>,
    registry: Arc<ModelRegistry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    source: String,
    revision: Option<String>,
    lora: Option<String>,
    profile: Option<String>,
}

pub(super) fn router(jobs: Arc<Jobs>, registry: Arc<ModelRegistry>) -> Router {
    Router::new()
        .route("/studio/api/models/{name}", put(install).delete(remove))
        .with_state(Arc::new(Management { jobs, registry }))
}

async fn install(
    State(state): State<Arc<Management>>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(request): Json<Registration>,
) -> Response {
    if !authorized(&state.jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    change(state, name, Some(request)).await
}

async fn remove(
    State(state): State<Arc<Management>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&state.jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    change(state, name, None).await
}

async fn change(state: Arc<Management>, name: String, request: Option<Registration>) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        validate_name(&name)?;
        let entry = request
            .map(|request| {
                let profile = request
                    .profile
                    .as_deref()
                    .map(|name| profiles::load(&state.jobs, name))
                    .transpose()?;
                if profile
                    .as_ref()
                    .is_some_and(|entry| entry.source != request.source)
                {
                    return Err("profile belongs to a different Studio model artifact".to_owned());
                }
                Ok::<_, String>(ModelEntry {
                    path: state.jobs.source(&request.source)?,
                    revision: request.revision,
                    lora: request
                        .lora
                        .as_deref()
                        .map(|name| state.jobs.adapter(name))
                        .transpose()?,
                    profile_artifact_sha256: profile
                        .as_ref()
                        .map(|profile| profile.artifact_sha256.clone()),
                    profile: profile.map(|profile| profile.settings),
                })
            })
            .transpose()?;
        state.registry.change_managed(name, entry)
    })
    .await;
    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(error)) => failure(StatusCode::BAD_REQUEST, &error),
        Err(error) => failure(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()),
    }
}
