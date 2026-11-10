//! Request admission and round-robin decode scheduling.

use std::collections::VecDeque;
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
        let request_id = request.id;
        let event = if request.timed_out() {
            request.timeout_event()
        } else {
            let step = request.session.step();
            if request.timed_out() {
                request.timeout_event()
            } else {
                match step {
                    Ok(Some(token_id)) => DecodeEvent::Token {
                        request_id,
                        token_id,
                        stop_reason: request.session.stop_reason(),
                    },
                    Ok(None) => DecodeEvent::Stopped {
                        request_id,
                        reason: request.session.stop_reason().expect("session finished"),
                    },
                    Err(error) => DecodeEvent::Failed { request_id, error },
                }
            }
        };
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
        Some(event)
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
}
