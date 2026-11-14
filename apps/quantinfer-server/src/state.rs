//! Shared HTTP server state.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::engine::{Metrics, Submission};
use crate::rate::RateLimit;
use crate::registry::ModelRegistry;

pub(crate) struct AppState {
    pub(crate) models: Arc<ModelRegistry>,
    pub(crate) queue: mpsc::Sender<Submission>,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) context_length: usize,
    pub(crate) request_timeout: Duration,
    pub(crate) rate_limit: Option<Mutex<RateLimit>>,
}
