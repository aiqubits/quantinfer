use super::*;
use crate::AwqOptions;
use half::f16;

fn inputs() -> (Tensor, Tensor) {
    let weight = Tensor::from_f32(
        vec![96, 2],
        (0..192)
            .map(|i| {
                if i / 2 % 32 == 31 {
                    9.1 + (i % 2) as f32
                } else {
                    ((i * 17) % 41) as f32 * 0.019 - 0.37
                }
            })
            .collect(),
    )
    .unwrap();
    let activations = Tensor::from_f32(
        vec![23, 96],
        (0..23 * 96)
            .map(|i| {
                let row = i / 96;
                let k = i % 96;
                let correlated = ((row * 13 + k / 4 * 3) % 37) as f32 * 0.03 - 0.5;
                correlated * if k % 32 == 31 { 0.001 } else { 1.0 }
            })
            .collect(),
    )
    .unwrap();
    (weight, activations)
}

// Literal token-row dot products with a separately constructed final F16 grid.
// This deliberately does not use the production covariance or packing helpers.
fn loss(
    weight: &[f32],
    x: &[f32],
    start: usize,
    end: usize,
    output: usize,
    bound: f32,
    bits: usize,
) -> f64 {
    let max = (1_i32 << (bits - 1)) - 1;
    let min = -(1_i32 << (bits - 1));
    let scale = if bound == 0. {
        1.
    } else {
        f16::from_f64(f64::from(bound) / f64::from(max))
            .to_f32()
            .max(f16::from_bits(1).to_f32())
    };
    let mut sum = 0.;
    for row in x.as_chunks::<96>().0 {
        let mut dot = 0.;
        for k in start..end {
            let original = weight[k * 2 + output];
            let code = (f64::from(original.clamp(-bound, bound)) / f64::from(scale))
                .round_ties_even()
                .clamp(f64::from(min), f64::from(max)) as i32;
            let decoded = scale * code as f32;
            dot += f64::from(row[k]) * (f64::from(decoded) - f64::from(original));
        }
        sum += dot * dot;
    }
    sum / 23.
}

#[test]
fn clipping_matches_literal_activation_loss_and_final_native_codes() {
    let (weight, activations) = inputs();
    let original = weight.as_f32().unwrap().to_vec();
    for bits in [4, 8] {
        for group_size in [32, 64] {
            let provider = AwqProvider::new(AwqOptions {
                bits,
                group_size,
                ..Default::default()
            })
            .unwrap();
            let plan = provider.search_clipping(&weight, &activations).unwrap();
            assert_eq!(plan.calibration_rows, 23);
            let groups = 96_usize.div_ceil(group_size);
            assert_eq!(plan.bounds.len(), groups * 2);
            let mut before = 0.;
            let mut after = 0.;
            for output in 0..2 {
                for (group, start) in (0..96).step_by(group_size).enumerate() {
                    let end = (start + group_size).min(96);
                    let maximum = (start..end)
                        .map(|i| original[i * 2 + output].abs())
                        .fold(0., f32::max);
                    let mut best = (f64::INFINITY, maximum);
                    for index in 0..10 {
                        let bound = maximum * (1. - index as f32 / 20.);
                        let error = loss(
                            &original,
                            activations.as_f32().unwrap(),
                            start,
                            end,
                            output,
                            bound,
                            bits,
                        );
                        if index == 0 {
                            before += error;
                        }
                        if error < best.0 {
                            best = (error, bound);
                        }
                    }
                    assert_eq!(plan.bounds[output * groups + group], best.1);
                    after += best.0;
                    for input in start..end {
                        assert_eq!(
                            plan.weights.as_f32().unwrap()[input * 2 + output],
                            original[input * 2 + output].clamp(-best.1, best.1)
                        );
                    }
                }
            }
            assert!((plan.group_loss_before - before).abs() < 1e-9 * (1. + before));
            assert!((plan.group_loss_after - after).abs() < 1e-9 * (1. + after));
            assert!(
                after < before,
                "bits={bits} group={group_size}: {before} -> {after}"
            );
            let packed = provider.quantize_transformed(&plan.weights).unwrap();
            let decoded = provider.dequantize(&packed).unwrap();
            let mut actual_loss = 0.;
            for output in 0..2 {
                for start in (0..96).step_by(group_size) {
                    let end = (start + group_size).min(96);
                    for row in activations.as_f32().unwrap().as_chunks::<96>().0 {
                        let error: f64 = (start..end)
                            .map(|k| {
                                f64::from(row[k])
                                    * (f64::from(decoded.as_f32().unwrap()[k * 2 + output])
                                        - f64::from(original[k * 2 + output]))
                            })
                            .sum();
                        actual_loss += error * error / 23.;
                    }
                }
            }
            assert!((actual_loss - after).abs() < 1e-9 * (1. + after));
            assert_eq!(weight.as_f32().unwrap(), original);
        }
    }
}

#[test]
fn clipping_rejects_invalid_input_and_insufficient_workspace() {
    let (weight, x) = inputs();
    let provider = AwqProvider::new(AwqOptions {
        maximum_workspace_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(provider
        .search_clipping(&weight, &x)
        .unwrap_err()
        .to_string()
        .contains("workspace"));
    let provider = AwqProvider::default();
    for invalid in [
        Tensor::from_f32(vec![1, 96], vec![f32::NAN; 96]).unwrap(),
        Tensor::from_f32(vec![1, 32], vec![1.; 32]).unwrap(),
        Tensor::from_f32(vec![96], vec![1.; 96]).unwrap(),
    ] {
        assert!(provider.search_clipping(&weight, &invalid).is_err());
    }
}
