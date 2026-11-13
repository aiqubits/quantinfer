//! Optional decode-phase timing.
//!
//! Timing is off until [`begin_phase_profile`]. Callers then pay for an
//! `Instant` around each matmul and attention call on this thread. Other
//! threads started inside those calls are included because the caller waits.

use std::cell::Cell;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseTotals {
    pub matmul_ns: u64,
    pub attention_ns: u64,
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TOTALS: Cell<PhaseTotals> = const { Cell::new(PhaseTotals { matmul_ns: 0, attention_ns: 0 }) };
}

pub fn begin_phase_profile() {
    TOTALS.set(PhaseTotals::default());
    ACTIVE.set(true);
}

pub fn end_phase_profile() -> Option<PhaseTotals> {
    if !ACTIVE.get() {
        return None;
    }
    ACTIVE.set(false);
    Some(TOTALS.get())
}

pub fn time_matmul<T>(body: impl FnOnce() -> T) -> T {
    time(true, body)
}

pub fn time_attention<T>(body: impl FnOnce() -> T) -> T {
    time(false, body)
}

fn time<T>(matmul: bool, body: impl FnOnce() -> T) -> T {
    if !ACTIVE.get() {
        return body();
    }
    let started = Instant::now();
    let value = body();
    let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    TOTALS.with(|totals| {
        let mut current = totals.get();
        if matmul {
            current.matmul_ns = current.matmul_ns.saturating_add(elapsed);
        } else {
            current.attention_ns = current.attention_ns.saturating_add(elapsed);
        }
        totals.set(current);
    });
    value
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{begin_phase_profile, end_phase_profile, time_attention, time_matmul};

    #[test]
    fn inactive_profile_records_nothing() {
        assert!(end_phase_profile().is_none());
        assert_eq!(time_matmul(|| 7), 7);
        assert!(end_phase_profile().is_none());
    }

    #[test]
    fn active_profile_attributes_matmul_and_attention() {
        begin_phase_profile();
        time_matmul(|| std::thread::sleep(Duration::from_millis(2)));
        time_attention(|| std::thread::sleep(Duration::from_millis(1)));
        let totals = end_phase_profile().expect("profile was active");
        assert!(totals.matmul_ns >= 1_000_000, "{}", totals.matmul_ns);
        assert!(totals.attention_ns >= 500_000, "{}", totals.attention_ns);
        assert!(end_phase_profile().is_none());
    }
}
