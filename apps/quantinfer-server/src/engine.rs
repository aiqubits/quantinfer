use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use quantinfer::cpu::CpuModel;
use quantinfer::runtime::{DecodeEvent, DecodeScheduler, DecodeSession};
use tokio::sync::{mpsc, oneshot};

#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub rejected: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub finished: AtomicU64,
    pub batched_steps: AtomicU64,
}

#[derive(Clone, Copy)]
pub struct BatchLimits {
    pub sequences: NonZeroUsize,
    pub tokens: NonZeroUsize,
}

pub struct Submission {
    pub session: DecodeSession<Arc<CpuModel>>,
    pub deadline: Instant,
    pub accepted: oneshot::Sender<Result<u64, String>>,
    pub output: mpsc::Sender<DecodeEvent>,
}

pub fn start(
    maximum_requests: usize,
    maximum_kv_bytes: usize,
    batch: BatchLimits,
    metrics: Arc<Metrics>,
) -> Result<(mpsc::Sender<Submission>, JoinHandle<()>), String> {
    let scheduler = DecodeScheduler::new(maximum_requests, maximum_kv_bytes)
        .map_err(|error| error.to_string())?;
    let (sender, receiver) = mpsc::channel(maximum_requests);
    let worker = thread::Builder::new()
        .name("quantinfer-decode".to_owned())
        .spawn(move || run(scheduler, receiver, maximum_requests, batch, &metrics))
        .map_err(|error| format!("cannot start decode worker: {error}"))?;
    Ok((sender, worker))
}

fn run(
    mut scheduler: DecodeScheduler<Arc<CpuModel>>,
    mut input: mpsc::Receiver<Submission>,
    maximum_requests: usize,
    batch: BatchLimits,
    metrics: &Metrics,
) {
    let mut listeners = BTreeMap::new();
    loop {
        if scheduler.pending_requests() == 0 {
            let Some(submission) = input.blocking_recv() else {
                return;
            };
            admit(submission, &mut scheduler, &mut listeners);
        }
        for _ in 0..maximum_requests {
            let Ok(submission) = input.try_recv() else {
                break;
            };
            admit(submission, &mut scheduler, &mut listeners);
        }
        let disconnected: Vec<_> = listeners
            .iter()
            .filter_map(|(&id, output)| output.is_closed().then_some(id))
            .collect();
        for id in disconnected {
            scheduler.cancel(id);
            listeners.remove(&id);
            metrics.finished.fetch_add(1, Ordering::Relaxed);
        }
        let events = scheduler.poll_batch(batch.sequences, batch.tokens);
        if events.len() > 1 {
            metrics.batched_steps.fetch_add(1, Ordering::Relaxed);
        }
        for event in events {
            let (id, finished) = match &event {
                DecodeEvent::Token {
                    request_id,
                    stop_reason,
                    ..
                } => {
                    metrics.generated_tokens.fetch_add(1, Ordering::Relaxed);
                    (*request_id, stop_reason.is_some())
                }
                DecodeEvent::Stopped { request_id, .. }
                | DecodeEvent::Failed { request_id, .. } => (*request_id, true),
            };
            let delivered = listeners
                .get(&id)
                .is_some_and(|output| output.try_send(event).is_ok());
            if finished || !delivered {
                if !finished {
                    scheduler.cancel(id);
                }
                listeners.remove(&id);
                metrics.finished.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn admit(
    submission: Submission,
    scheduler: &mut DecodeScheduler<Arc<CpuModel>>,
    listeners: &mut BTreeMap<u64, mpsc::Sender<DecodeEvent>>,
) {
    let remaining = submission
        .deadline
        .saturating_duration_since(Instant::now());
    let result = scheduler
        .submit(submission.session, Some(remaining))
        .map_err(|error| error.to_string());
    match result {
        Ok(id) => {
            if submission.accepted.send(Ok(id)).is_ok() {
                listeners.insert(id, submission.output);
            } else {
                scheduler.cancel(id);
            }
        }
        Err(error) => {
            let _ = submission.accepted.send(Err(error));
        }
    }
}
