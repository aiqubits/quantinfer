//! Request admission and round-robin decode scheduling.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{DecodeSession, DecodeStopReason, DecoderModel, Result, RuntimeError};

#[derive(Debug, Clone, PartialEq)]
pub enum DecodeEvent {
    Token {
        request_id: u64,
        token_id: usize,
        stop_reason: Option<DecodeStopReason>,
    },
    Stopped {
        request_id: u64,
        reason: DecodeStopReason,
    },
    Failed {
        request_id: u64,
        error: RuntimeError,
    },
}

struct QueuedDecode<M> {
    id: u64,
    session: DecodeSession<M>,
    deadline: Option<Instant>,
}

impl<M> QueuedDecode<M> {
    fn timed_out(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn timeout_event(&self) -> DecodeEvent {
        DecodeEvent::Stopped {
            request_id: self.id,
            reason: DecodeStopReason::TimedOut,
        }
    }
}

impl<M: DecoderModel> QueuedDecode<M> {
    fn step(&mut self) -> DecodeEvent {
        if self.timed_out() {
            return self.timeout_event();
        }
        let step = self.session.step();
        self.event_after_step(step)
    }

    fn event_after_step(&self, step: Result<Option<usize>>) -> DecodeEvent {
        if self.timed_out() {
            return self.timeout_event();
        }
        let request_id = self.id;
        match step {
            Ok(Some(token_id)) => DecodeEvent::Token {
                request_id,
                token_id,
                stop_reason: self.session.stop_reason(),
            },
            Ok(None) => DecodeEvent::Stopped {
                request_id,
                reason: self.session.stop_reason().expect("session finished"),
            },
            Err(error) => DecodeEvent::Failed { request_id, error },
        }
    }
}

/// Reserves KV capacity at admission and advances one runnable request per poll.
pub struct DecodeScheduler<M> {
    ready: VecDeque<QueuedDecode<M>>,
    next_id: u64,
    maximum_requests: usize,
    maximum_kv_bytes: usize,
    reserved_kv_bytes: usize,
}

impl<M: DecoderModel> DecodeScheduler<M> {
    pub fn new(maximum_requests: usize, maximum_kv_bytes: usize) -> Result<Self> {
        if maximum_requests == 0 || maximum_kv_bytes == 0 {
            return Err(RuntimeError::InvalidInput(
                "scheduler quotas must be positive".to_owned(),
            ));
        }
        Ok(Self {
            ready: VecDeque::new(),
            next_id: 1,
            maximum_requests,
            maximum_kv_bytes,
            reserved_kv_bytes: 0,
        })
    }

    pub fn pending_requests(&self) -> usize {
        self.ready.len()
    }

    pub fn reserved_kv_bytes(&self) -> usize {
        self.reserved_kv_bytes
    }

    pub fn submit(&mut self, session: DecodeSession<M>, timeout: Option<Duration>) -> Result<u64> {
        if session.stop_reason().is_some() {
            return Err(RuntimeError::InvalidInput(
                "cannot queue a stopped request".to_owned(),
            ));
        }
        if self.ready.len() >= self.maximum_requests {
            return Err(RuntimeError::InvalidInput(
                "scheduler request quota exceeded".to_owned(),
            ));
        }
        let reserved = self
            .reserved_kv_bytes
            .checked_add(session.cache().capacity_bytes())
            .filter(|bytes| *bytes <= self.maximum_kv_bytes)
            .ok_or_else(|| RuntimeError::InvalidInput("scheduler KV quota exceeded".to_owned()))?;
        let deadline = timeout
            .map(|duration| {
                Instant::now().checked_add(duration).ok_or_else(|| {
                    RuntimeError::InvalidInput("scheduler deadline overflows the clock".to_owned())
                })
            })
            .transpose()?;
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or_else(|| {
            RuntimeError::InvalidInput("scheduler request IDs are exhausted".to_owned())
        })?;
        self.ready.push_back(QueuedDecode {
            id,
            session,
            deadline,
        });
        self.reserved_kv_bytes = reserved;
        Ok(id)
    }

    pub fn cancel(&mut self, request_id: u64) -> Option<DecodeEvent> {
        let index = self
            .ready
            .iter()
            .position(|request| request.id == request_id)?;
        let request = self.ready.remove(index)?;
        self.reserved_kv_bytes -= request.session.cache().capacity_bytes();
        Some(DecodeEvent::Stopped {
            request_id,
            reason: DecodeStopReason::Cancelled,
        })
    }

    /// Check the timeout before and after the decoder's indivisible forward pass.
    pub fn poll_next(&mut self) -> Option<DecodeEvent> {
        let mut request = self.ready.pop_front()?;
        let event = request.step();
        Some(self.finish_poll(request, event))
    }

    fn finish_poll(&mut self, request: QueuedDecode<M>, event: DecodeEvent) -> DecodeEvent {
        if matches!(
            event,
            DecodeEvent::Token {
                stop_reason: None,
                ..
            }
        ) {
            self.ready.push_back(request);
        } else {
            self.reserved_kv_bytes -= request.session.cache().capacity_bytes();
        }
        event
    }
}

impl<M: DecoderModel> DecodeScheduler<Arc<M>> {
    /// Batch ready requests for the oldest request's model. A prompt larger than
    /// the batch token budget receives an exclusive step.
    pub fn poll_batch(
        &mut self,
        max_sequences: NonZeroUsize,
        max_tokens: NonZeroUsize,
    ) -> Vec<DecodeEvent> {
        let Some(first) = self.ready.front() else {
            return Vec::new();
        };
        let mut remaining = max_tokens
            .get()
            .saturating_sub(first.session.pending_tokens());
        if first.timed_out() || max_sequences.get() == 1 || remaining == 0 {
            return self.poll_next().into_iter().collect();
        }
        let mut batch = vec![self.ready.pop_front().expect("non-empty queue")];
        let mut index = 0;
        while batch.len() < max_sequences.get() && remaining > 0 && index < self.ready.len() {
            let request = &self.ready[index];
            let tokens = request.session.pending_tokens();
            if tokens <= remaining
                && !request.timed_out()
                && Arc::ptr_eq(request.session.model(), batch[0].session.model())
            {
                remaining -= tokens;
                batch.push(self.ready.remove(index).expect("selected ready request"));
            } else {
                index += 1;
            }
        }
        if batch.len() == 1 {
            let mut request = batch.pop().expect("one ready request");
            let event = request.step();
            return vec![self.finish_poll(request, event)];
        }
        let mut sessions: Vec<_> = batch
            .iter_mut()
            .map(|request| &mut request.session)
            .collect();
        let outputs = DecodeSession::step_batch(&mut sessions);
        batch
            .into_iter()
            .zip(outputs)
            .map(|(request, step)| {
                let event = request.event_after_step(step.map(Some));
                self.finish_poll(request, event)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use quantinfer_core::Tensor;

    use crate::{GenerationConfig, KvCache};

    use super::*;

    struct Fixed;

    impl DecoderModel for Fixed {
        fn forward(&self, _tokens: &[usize], _cache: &mut KvCache) -> Result<Tensor> {
            Ok(Tensor::from_f32(vec![1, 2], vec![0.0, 1.0])?)
        }
    }

    fn session(context: usize) -> DecodeSession<Fixed> {
        DecodeSession::new(
            Fixed,
            KvCache::new(1, context, 1, 1).unwrap(),
            vec![0],
            GenerationConfig {
                max_new_tokens: 2,
                ..GenerationConfig::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn schedules_requests_in_order_and_releases_their_quota() {
        let mut scheduler = DecodeScheduler::new(2, 64).unwrap();
        let first = scheduler.submit(session(4), None).unwrap();
        let second = scheduler.submit(session(4), None).unwrap();
        assert_eq!(scheduler.reserved_kv_bytes(), 64);
        assert!(scheduler.submit(session(4), None).is_err());
        for (request_id, stop_reason) in [
            (first, None),
            (second, None),
            (first, Some(DecodeStopReason::MaxTokens)),
        ] {
            assert_eq!(
                scheduler.poll_next(),
                Some(DecodeEvent::Token {
                    request_id,
                    token_id: 1,
                    stop_reason
                })
            );
        }
        assert_eq!(scheduler.reserved_kv_bytes(), 32);
        assert_eq!(
            scheduler.cancel(second),
            Some(DecodeEvent::Stopped {
                request_id: second,
                reason: DecodeStopReason::Cancelled,
            })
        );
        assert_eq!(scheduler.reserved_kv_bytes(), 0);
        assert_eq!(scheduler.poll_next(), None);
    }

    #[test]
    fn enforces_kv_quota_and_expires_requests_before_decoding() {
        let mut scheduler = DecodeScheduler::new(2, 32).unwrap();
        let request_id = scheduler.submit(session(4), Some(Duration::ZERO)).unwrap();
        assert!(scheduler.submit(session(4), None).is_err());
        assert_eq!(
            scheduler.poll_next(),
            Some(DecodeEvent::Stopped {
                request_id,
                reason: DecodeStopReason::TimedOut,
            })
        );
        assert_eq!(scheduler.pending_requests(), 0);
        assert_eq!(scheduler.reserved_kv_bytes(), 0);
    }

    #[test]
    fn batches_one_model_within_budget_and_admits_new_requests_between_steps() {
        #[derive(Default)]
        struct Counting(std::sync::Mutex<Vec<Vec<usize>>>);
        impl DecoderModel for Counting {
            fn forward(&self, _: &[usize], _: &mut KvCache) -> Result<Tensor> {
                Ok(Tensor::from_f32(vec![1, 2], vec![0.0, 1.0])?)
            }

            fn forward_batch(
                &self,
                batch: &mut [crate::DecodeBatchItem<'_>],
            ) -> Vec<Result<Tensor>> {
                self.0
                    .lock()
                    .unwrap()
                    .push(batch.iter().map(|item| item.tokens.len()).collect());
                batch
                    .iter()
                    .map(|_| Ok(Tensor::from_f32(vec![2], vec![0.0, 1.0])?))
                    .collect()
            }
        }
        let session = |model: &Arc<Counting>, prompt_len| {
            DecodeSession::new(
                model.clone(),
                KvCache::new(1, 8, 1, 1).unwrap(),
                vec![0; prompt_len],
                GenerationConfig {
                    max_new_tokens: 2,
                    ..GenerationConfig::default()
                },
            )
            .unwrap()
        };
        let model = Arc::new(Counting::default());
        let other = Arc::new(Counting::default());
        let mut scheduler = DecodeScheduler::new(5, 320).unwrap();
        let first = scheduler.submit(session(&model, 2), None).unwrap();
        let different = scheduler.submit(session(&other, 1), None).unwrap();
        let peer = scheduler.submit(session(&model, 1), None).unwrap();
        let oversized = scheduler.submit(session(&model, 4), None).unwrap();
        let ids = |events: Vec<DecodeEvent>| {
            events
                .into_iter()
                .map(|event| match event {
                    DecodeEvent::Token {
                        request_id,
                        token_id: 1,
                        ..
                    } => request_id,
                    other => panic!("unexpected event {other:?}"),
                })
                .collect::<Vec<_>>()
        };
        let width = NonZeroUsize::new(3).unwrap();
        assert_eq!(ids(scheduler.poll_batch(width, width)), [first, peer]);
        assert_eq!(ids(scheduler.poll_batch(width, width)), [different]);
        assert_eq!(ids(scheduler.poll_batch(width, width)), [oversized]);
        assert!(scheduler.cancel(first).is_some());
        let newcomer = scheduler.submit(session(&model, 2), None).unwrap();
        assert_eq!(ids(scheduler.poll_batch(width, width)), [peer, oversized]);
        assert_eq!(ids(scheduler.poll_batch(width, width)), [different]);
        assert_eq!(ids(scheduler.poll_batch(width, width)), [newcomer]);
        assert_eq!(*model.0.lock().unwrap(), [vec![2, 1], vec![1, 1]]);
        assert!(other.0.lock().unwrap().is_empty());
        assert_eq!(scheduler.reserved_kv_bytes(), 64);
    }
}
