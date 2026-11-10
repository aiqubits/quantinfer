use std::num::NonZeroUsize;
use std::time::Instant;

pub struct RateLimit {
    per_second: f64,
    available: f64,
    last_refill: Instant,
}

impl RateLimit {
    pub fn new(requests_per_second: NonZeroUsize, now: Instant) -> Self {
        let per_second = requests_per_second.get() as f64;
        Self {
            per_second,
            available: per_second,
            last_refill: now,
        }
    }

    pub fn allow(&mut self, now: Instant) -> bool {
        let elapsed = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        self.last_refill = now;
        self.available = (self.available + elapsed * self.per_second).min(self.per_second);
        if self.available < 1.0 {
            return false;
        }
        self.available -= 1.0;
        true
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn refills_up_to_one_second_of_burst_capacity() {
        let now = Instant::now();
        let mut limit = RateLimit::new(NonZeroUsize::new(2).unwrap(), now);
        assert!(limit.allow(now));
        assert!(limit.allow(now));
        assert!(!limit.allow(now));
        assert!(!limit.allow(now + Duration::from_millis(250)));
        assert!(limit.allow(now + Duration::from_millis(500)));
        assert!(!limit.allow(now + Duration::from_millis(500)));
        let later = now + Duration::from_secs(10);
        assert!(limit.allow(later));
        assert!(limit.allow(later));
        assert!(!limit.allow(later));
    }
}
