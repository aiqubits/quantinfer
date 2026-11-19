//! Generation and streaming HTTP routes.

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use quantinfer::runtime::{DecodeEvent, DecodeSession, DecodeStopReason, GenerationConfig};
use quantinfer::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::engine::Submission;
use crate::state::AppState;

use super::{error_response, ApiError};

#[derive(Deserialize)]
pub(crate) struct GenerateRequest {
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

pub(crate) async fn generate(
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
    let context = state.context_length.min(model.context_length());
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
