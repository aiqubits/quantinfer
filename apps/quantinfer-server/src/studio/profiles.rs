//! Artifact-bound CPU profiles for use by managed registrations.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use quantinfer::{cpu::ModelSource, runtime::InferenceProfile};
use serde::{Deserialize, Serialize};

use super::{
    authorized, failure,
    jobs::{read_json, validate_name},
    Jobs,
};
use crate::registry::{artifact_sha256, validated_profile, CPU_SCALAR_PROVIDER};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    source: String,
    model_variant: Option<String>,
    context_length: usize,
    thread_count: usize,
    memory_budget_bytes: Option<usize>,
    quality_threshold: Option<f32>,
    performance_baseline_tokens_per_second: Option<f64>,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Profile {
    pub source: String,
    pub artifact_sha256: String,
    pub settings: InferenceProfile,
}

pub(super) fn router(jobs: Arc<Jobs>) -> Router {
    Router::new()
        .route(
            "/studio/api/profiles/{name}",
            get(show).put(save).delete(remove),
        )
        .with_state(jobs)
}

pub(super) fn load(jobs: &Jobs, name: &str) -> Result<Profile, String> {
    let path = profile_path(jobs, name)?;
    serde_json::from_value(read_json(&path, 64 * 1024)?).map_err(display)
}

fn profile_path(jobs: &Jobs, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    if name.len() > 110 {
        return Err("profile name is too long".into());
    }
    Ok(jobs.profile_directory().join(format!("{name}.json")))
}

async fn show(
    State(jobs): State<Arc<Jobs>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&jobs, &headers, false) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match load(&jobs, &name) {
        Ok(profile) => Json(profile).into_response(),
        Err(error) => failure(StatusCode::NOT_FOUND, &error),
    }
}

async fn save(
    State(jobs): State<Arc<Jobs>>,
    Path(name): Path<String>,
    headers: HeaderMap,
    Json(settings): Json<Settings>,
) -> Response {
    if !authorized(&jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match tokio::task::spawn_blocking(move || save_profile(&jobs, &name, settings)).await {
        Ok(Ok(profile)) => Json(profile).into_response(),
        Ok(Err(error)) => failure(StatusCode::BAD_REQUEST, &error),
        Err(error) => failure(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()),
    }
}

fn save_profile(jobs: &Jobs, name: &str, options: Settings) -> Result<Profile, String> {
    let path = profile_path(jobs, name)?;
    if !(1..=64).contains(&options.thread_count) {
        return Err("profile threads must be between 1 and 64".into());
    }
    let artifact = jobs.source(&options.source)?;
    let sha256 = artifact_sha256(&artifact)?;
    let model = ModelSource::open(&artifact).map_err(display)?;
    let reader = model
        .open_reader(options.memory_budget_bytes)
        .map_err(display)?;
    let declared = model.package().map(|package| &package.manifest);
    let manifest = declared
        .or(reader.embedded_manifest())
        .ok_or("profile source requires a model manifest")?;
    let variant = options.model_variant.unwrap_or_else(|| {
        manifest
            .execution_variants
            .iter()
            .find(|variant| {
                variant.runtime_provider == CPU_SCALAR_PROVIDER
                    && variant.kernel_provider == CPU_SCALAR_PROVIDER
            })
            .map_or_else(|| "cpu".to_owned(), |variant| variant.name.clone())
    });
    let settings = InferenceProfile {
        model_variant: variant,
        quantization_variant: manifest.quantization.name().into(),
        runtime_provider: CPU_SCALAR_PROVIDER.into(),
        kernel_provider: CPU_SCALAR_PROVIDER.into(),
        context_length: options.context_length,
        thread_count: options.thread_count,
        memory_budget_bytes: options.memory_budget_bytes,
        quality_threshold: options.quality_threshold,
        performance_baseline_tokens_per_second: options.performance_baseline_tokens_per_second,
    };
    validated_profile(&reader, declared, &settings)?;
    if artifact_sha256(&artifact)? != sha256 {
        return Err("profile source changed while it was checked".into());
    }
    let profile = Profile {
        source: options.source,
        artifact_sha256: sha256,
        settings,
    };
    let mut stage = tempfile::NamedTempFile::new_in(jobs.profile_directory()).map_err(display)?;
    serde_json::to_writer(&mut stage, &profile).map_err(display)?;
    stage.as_file().sync_all().map_err(display)?;
    stage.persist(path).map_err(display)?;
    Ok(profile)
}

async fn remove(
    State(jobs): State<Arc<Jobs>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&jobs, &headers, true) {
        return failure(
            StatusCode::FORBIDDEN,
            "Studio requires the local server origin",
        );
    }
    match profile_path(&jobs, &name).and_then(|path| fs::remove_file(path).map_err(display)) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => failure(StatusCode::BAD_REQUEST, &error),
    }
}

fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
