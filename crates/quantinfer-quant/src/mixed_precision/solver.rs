use std::collections::BTreeSet;

use quantinfer_core::QuantizationScheme;

use super::{OptimizedPrecisionPlan, PrecisionGroup, PrecisionSearchLimits};
use crate::{MixedPrecisionPlan, QuantizationError, Result};

#[derive(Clone, Copy)]
struct State {
    bytes: u64,
    error: f64,
    history: Option<usize>,
}

struct Decision {
    parent: Option<usize>,
    candidate: usize,
}

struct Pending {
    bytes: u64,
    error: f64,
    parent: Option<usize>,
    candidate: usize,
    ordinal: usize,
}

fn invalid(message: impl Into<String>) -> QuantizationError {
    QuantizationError::InvalidInput(message.into())
}

/// Minimize summed candidate error within a total encoded-weight byte budget.
///
/// `fixed_encoded_bytes` includes weights outside the selectable groups. Every
/// group is assigned exactly once. Equal objectives prefer fewer bytes, then
/// stable input traversal order. Optimality is relative to the supplied finite
/// F64 measurements, not a claim about model task accuracy. No files are parsed.
pub fn optimize_mixed_precision(
    groups: &[PrecisionGroup],
    fixed_encoded_bytes: u64,
    maximum_encoded_bytes: u64,
    limits: PrecisionSearchLimits,
) -> Result<OptimizedPrecisionPlan> {
    if groups.is_empty() || groups.len() > 4096 {
        return Err(invalid("precision search requires 1..=4096 groups"));
    }
    if limits.max_frontier_states == 0
        || limits.max_frontier_states > 1_048_576
        || limits.max_history_states == 0
        || limits.max_history_states > 4_194_304
    {
        return Err(invalid(
            "precision search limits are outside supported bounds",
        ));
    }
    if fixed_encoded_bytes > maximum_encoded_bytes {
        return Err(invalid("fixed encoded weights exceed the precision budget"));
    }
    let mut names = BTreeSet::new();
    for group in groups {
        if group.tensor_names.is_empty()
            || group.candidates.is_empty()
            || group.candidates.len() > 6
        {
            return Err(invalid("precision group needs names and 1..=6 candidates"));
        }
        for name in &group.tensor_names {
            if name.is_empty() || !names.insert(name) {
                return Err(invalid(
                    "precision tensor names must be unique and nonempty",
                ));
            }
            if names.len() > 1_048_576 {
                return Err(invalid("precision tensor name count exceeds the limit"));
            }
        }
        let mut schemes = Vec::new();
        for candidate in &group.candidates {
            if candidate.scheme == QuantizationScheme::Mixed
                || schemes.contains(&candidate.scheme)
                || candidate.encoded_bytes == 0
                || !candidate.error.is_finite()
                || candidate.error < 0.0
            {
                return Err(invalid("invalid or duplicate precision candidate"));
            }
            schemes.push(candidate.scheme);
        }
    }
    let mut frontier = vec![State {
        bytes: fixed_encoded_bytes,
        error: 0.0,
        history: None,
    }];
    let mut history = Vec::<Decision>::new();
    for group in groups {
        let capacity = frontier
            .len()
            .checked_mul(group.candidates.len())
            .ok_or_else(|| invalid("precision expansion overflows usize"))?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(capacity)
            .map_err(|error| invalid(format!("cannot allocate precision expansion: {error}")))?;
        for (state_index, state) in frontier.iter().enumerate() {
            for (candidate_index, candidate) in group.candidates.iter().enumerate() {
                let Some(bytes) = state.bytes.checked_add(candidate.encoded_bytes) else {
                    // Overflow exceeds every representable u64 budget.
                    continue;
                };
                if bytes > maximum_encoded_bytes {
                    continue;
                }
                let error = state.error + candidate.error;
                if !error.is_finite() {
                    return Err(invalid("summed precision error exceeds finite F64 range"));
                }
                pending.push(Pending {
                    bytes,
                    error,
                    parent: state.history,
                    candidate: candidate_index,
                    ordinal: state_index * group.candidates.len() + candidate_index,
                });
            }
        }
        if pending.is_empty() {
            return Err(invalid(
                "no precision assignment fits the encoded-weight budget",
            ));
        }
        pending.sort_unstable_by(|a, b| {
            a.bytes
                .cmp(&b.bytes)
                .then_with(|| a.error.total_cmp(&b.error))
                .then_with(|| a.ordinal.cmp(&b.ordinal))
        });
        frontier.clear();
        let mut best_error = f64::INFINITY;
        for state in pending {
            if state.error >= best_error {
                continue;
            }
            if frontier.len() >= limits.max_frontier_states {
                return Err(invalid(format!(
                    "exact precision search exceeded frontier state limit {} at {}",
                    limits.max_frontier_states, group.tensor_names[0]
                )));
            }
            if history.len() >= limits.max_history_states {
                return Err(invalid(format!(
                    "exact precision search exceeded history state limit {} at {}",
                    limits.max_history_states, group.tensor_names[0]
                )));
            }
            history
                .try_reserve(1)
                .map_err(|error| invalid(format!("cannot allocate precision history: {error}")))?;
            frontier
                .try_reserve(1)
                .map_err(|error| invalid(format!("cannot allocate precision frontier: {error}")))?;
            let index = history.len();
            history.push(Decision {
                parent: state.parent,
                candidate: state.candidate,
            });
            frontier.push(State {
                bytes: state.bytes,
                error: state.error,
                history: Some(index),
            });
            best_error = state.error;
        }
    }
    // Pareto errors strictly decrease as byte counts increase.
    let best = frontier.last().expect("nonempty feasible frontier");
    let mut trace = best.history;
    let mut plan = MixedPrecisionPlan::default();
    for group in groups.iter().rev() {
        let decision = &history[trace.expect("one decision per group")];
        for name in &group.tensor_names {
            plan.assign(name, group.candidates[decision.candidate].scheme);
        }
        trace = decision.parent;
    }
    Ok(OptimizedPrecisionPlan {
        plan,
        encoded_bytes: best.bytes,
        error: best.error,
    })
}
