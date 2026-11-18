//! Exact precision assignment for measured candidates under an encoded-weight budget.
//!
//! Candidate errors are supplied by calibration/reconstruction, not inferred
//! from bit counts. The solver minimizes their sum using a Pareto frontier.
//! Limits cause an explicit error, never an approximate success. Encoded tensor
//! bytes exclude container metadata, padding, compression and runtime workspace.

mod measure;
mod solver;

use quantinfer_core::QuantizationScheme;

use crate::MixedPrecisionPlan;

pub use measure::{
    measure_linear_precision_candidates, PrecisionMeasurements, UnavailablePrecision,
};
pub use solver::optimize_mixed_precision;

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionCandidate {
    pub scheme: QuantizationScheme,
    pub encoded_bytes: u64,
    /// Finite, nonnegative reconstruction/calibration objective.
    pub error: f64,
}

/// Names in one group share a choice and its encoded byte cost.
///
/// Callers defining alias groups must establish compatible payloads/layouts.
/// Their candidate error must include every logical use they want to weight.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionGroup {
    pub tensor_names: Vec<String>,
    pub candidates: Vec<PrecisionCandidate>,
}

#[derive(Debug, Clone, Copy)]
pub struct PrecisionSearchLimits {
    pub max_frontier_states: usize,
    pub max_history_states: usize,
}

impl Default for PrecisionSearchLimits {
    fn default() -> Self {
        Self {
            max_frontier_states: 65_536,
            max_history_states: 1_048_576,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizedPrecisionPlan {
    pub plan: MixedPrecisionPlan,
    pub encoded_bytes: u64,
    pub error: f64,
}

#[cfg(test)]
mod tests;
