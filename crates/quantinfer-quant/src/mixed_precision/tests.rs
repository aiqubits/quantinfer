use super::*;
use quantinfer_core::QuantizationScheme::{None as F32, Q4_0, Q8_0};

fn group(name: &str, choices: &[(QuantizationScheme, u64, f64)]) -> PrecisionGroup {
    PrecisionGroup {
        tensor_names: vec![name.into()],
        candidates: choices
            .iter()
            .map(|&(scheme, encoded_bytes, error)| PrecisionCandidate {
                scheme,
                encoded_bytes,
                error,
            })
            .collect(),
    }
}

#[test]
fn global_budget_beats_greedy_upgrades_and_counts_alias_groups_once() {
    let mut a = group("a", &[(Q4_0, 1, 9.0), (Q8_0, 3, 0.0)]);
    a.tensor_names.push("a_alias".into());
    let b = group("b", &[(Q4_0, 1, 6.0), (Q8_0, 2, 0.0)]);
    // Greedily taking B's higher gain/byte first leaves error 9. The optimum is 6.
    let solution = optimize_mixed_precision(&[a, b], 7, 11, Default::default()).unwrap();
    assert_eq!((solution.encoded_bytes, solution.error), (11, 6.0));
    assert_eq!(solution.plan.scheme_for("a"), Some(Q8_0));
    assert_eq!(solution.plan.scheme_for("a_alias"), Some(Q8_0));
    assert_eq!(solution.plan.scheme_for("b"), Some(Q4_0));
}

#[test]
fn exact_frontiers_match_independent_exhaustive_assignments() {
    fn enumerate(groups: &[PrecisionGroup], bytes: u64, error: f64, budget: u64) -> (f64, u64) {
        if bytes > budget {
            return (f64::INFINITY, u64::MAX);
        }
        let Some((group, rest)) = groups.split_first() else {
            return (error, bytes);
        };
        group
            .candidates
            .iter()
            .map(|c| enumerate(rest, bytes + c.encoded_bytes, error + c.error, budget))
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .unwrap()
    }
    let mut seed = 151_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed >> 32
    };
    for case in 0..96 {
        let groups = (0..1 + case % 6)
            .map(|index| {
                group(
                    &format!("tensor-{index}"),
                    &[
                        (Q4_0, 1 + next() % 20, (next() % 31) as f64),
                        (Q8_0, 1 + next() % 20, (next() % 31) as f64),
                        (F32, 1 + next() % 20, 0.0),
                    ],
                )
            })
            .collect::<Vec<_>>();
        for budget in [10, 25, 60, 150] {
            let expected = enumerate(&groups, 7, 0.0, budget);
            let actual = optimize_mixed_precision(&groups, 7, budget, Default::default());
            if expected.0.is_infinite() {
                assert!(actual.is_err());
            } else {
                let actual = actual.unwrap();
                assert_eq!((actual.error, actual.encoded_bytes), expected);
                let selected = groups
                    .iter()
                    .map(|g| {
                        g.candidates
                            .iter()
                            .find(|c| actual.plan.scheme_for(&g.tensor_names[0]) == Some(c.scheme))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    7 + selected.iter().map(|c| c.encoded_bytes).sum::<u64>(),
                    expected.1
                );
                assert_eq!(selected.iter().map(|c| c.error).sum::<f64>(), expected.0);
            }
        }
    }
}

#[test]
fn state_limits_fail_explicitly_instead_of_claiming_an_approximate_optimum() {
    let groups = [group("a", &[(Q4_0, 1, 3.0), (Q8_0, 2, 2.0), (F32, 3, 1.0)])];
    let limits = PrecisionSearchLimits {
        max_frontier_states: 2,
        ..Default::default()
    };
    let error = optimize_mixed_precision(&groups, 0, 3, limits).unwrap_err();
    assert!(error.to_string().contains("state limit"));
    let groups = [group("a", &[(F32, 1, 0.0)]), group("b", &[(F32, 1, 0.0)])];
    let limits = PrecisionSearchLimits {
        max_history_states: 1,
        ..Default::default()
    };
    assert!(optimize_mixed_precision(&groups, 0, 2, limits).is_err());
}

#[test]
fn deterministic_ties_and_overflowing_costs_are_handled() {
    let groups = [group("a", &[(Q4_0, 2, 0.0), (Q8_0, 1, 0.0), (F32, 1, 0.0)])];
    for _ in 0..3 {
        let result = optimize_mixed_precision(&groups, 0, 2, Default::default()).unwrap();
        assert_eq!(result.plan.scheme_for("a"), Some(Q8_0));
        assert_eq!(result.encoded_bytes, 1);
    }
    let groups = [group("a", &[(Q4_0, 2, 0.0), (Q8_0, 1, 1.0)])];
    let result =
        optimize_mixed_precision(&groups, u64::MAX - 1, u64::MAX, Default::default()).unwrap();
    assert_eq!(result.plan.scheme_for("a"), Some(Q8_0));
    assert_eq!(result.encoded_bytes, u64::MAX);
}

#[test]
fn rejects_invalid_objectives_names_and_empty_choices() {
    for error in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(optimize_mixed_precision(
            &[group("a", &[(Q4_0, 1, error)])],
            0,
            10,
            Default::default()
        )
        .is_err());
    }
    for groups in [
        vec![group("a", &[])],
        vec![group("", &[(Q4_0, 1, 1.0)])],
        vec![group("a", &[(Q4_0, 0, 1.0)])],
        vec![group("a", &[(QuantizationScheme::Mixed, 1, 1.0)])],
        vec![group("a", &[(Q4_0, 1, 1.0), (Q4_0, 2, 0.0)])],
        vec![group("a", &[(Q4_0, 1, 1.0)]), group("a", &[(Q8_0, 2, 0.0)])],
    ] {
        assert!(optimize_mixed_precision(&groups, 0, 10, Default::default()).is_err());
    }
}

#[test]
fn channel_measurements_use_stored_scales_and_report_unavailable_encodings() {
    let input = quantinfer_core::Tensor::from_f32(vec![2, 16], vec![1.0e-8; 32]).unwrap();
    let measured = measure_linear_precision_candidates(
        "weight",
        &input,
        &[
            Q4_0,
            Q8_0,
            F32,
            QuantizationScheme::Q4_K,
            QuantizationScheme::Q5_K,
            QuantizationScheme::Q6_K,
        ],
        Some(&[10.0, 0.0]),
    )
    .unwrap();
    assert_eq!(measured.unavailable.len(), 3);
    assert_eq!(measured.group.candidates.len(), 3);
    let expected = f64::from(1.0e-8_f32).powi(2) * 16.0 * 10.0;
    for candidate in &measured.group.candidates {
        match candidate.scheme {
            F32 => assert_eq!((candidate.encoded_bytes, candidate.error), (128, 0.0)),
            Q4_0 => {
                assert_eq!(candidate.encoded_bytes, 18);
                assert!((candidate.error - expected).abs() < 1.0e-28);
            }
            Q8_0 => {
                assert_eq!(candidate.encoded_bytes, 34);
                assert!((candidate.error - expected).abs() < 1.0e-28);
            }
            _ => unreachable!(),
        }
    }
    for weights in [&[1.0][..], &[0.0, 0.0], &[f32::NAN, 1.0], &[-1.0, 1.0]] {
        assert!(
            measure_linear_precision_candidates("weight", &input, &[Q4_0], Some(weights)).is_err()
        );
    }
}
