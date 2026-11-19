//! Embedded local dashboard and registry metadata route.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde_json::json;

use crate::state::AppState;

/// Serve a dependency-free local dashboard for model registry and inference
/// smoke tests. The page deliberately uses the public HTTP endpoints so it is
/// also useful as a small integration example for clients.
pub(crate) async fn studio() -> Html<&'static str> {
    Html(STUDIO_HTML)
}

pub(crate) async fn studio_registry(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({"models": state.models.descriptors()}))
}

pub(crate) const STUDIO_HTML: &str = concat!(
    include_str!("../studio/dashboard.html"),
    "<script>\n",
    include_str!("../studio/dashboard.js"),
    "</script>\n</body>\n</html>\n",
);
