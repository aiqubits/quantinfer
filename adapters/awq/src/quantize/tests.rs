use super::*;
use half::f16;

fn weights() -> Tensor {
    Tensor::from_f32(
        vec![64, 3],
        (0..192)
            .map(|i| ((i * 19 + i / 7) % 97) as f32 * 0.0053 - 0.24)
            .collect(),
    )
    .unwrap()
}
fn activations() -> Tensor {
    Tensor::from_f32(
        vec![48, 64],
        (0..48 * 64)
            .map(|i| {
                let k = i % 64;
                let base = ((i * 13 + i / 64 * 7) % 101) as f32 * 0.021 - 1.;
                base * if k % 13 == 0 {
                    30.
                } else {
                    0.25 + k as f32 * 0.01
                }
            })
            .collect(),
    )
    .unwrap()
}
fn output_error(weight: &Tensor, candidate: &Tensor, input: &Tensor) -> f64 {
    let (k, n) = (64, 3);
    let source = weight.as_f32().unwrap();
    let target = candidate.as_f32().unwrap();
    let x = input.as_f32().unwrap();
    let mut sum = 0.;
    for row in x.chunks_exact(k) {
        for o in 0..n {
            let difference: f64 = (0..k)
                .map(|i| {
                    f64::from(row[i])
                        * (f64::from(target[i * n + o]) - f64::from(source[i * n + o]))
                })
                .sum();
            sum += difference * difference;
        }
    }
    sum / (input.shape().dims()[0] * n) as f64
}

#[test]
fn scale_search_uses_activation_loss_and_improves_the_declared_linear_case() {
    let weight = weights();
    let input = activations();
    let provider = AwqProvider::default();
    let mut calibration = AwqCalibration::new(64, 512).unwrap();
    calibration.observe_rows(&input).unwrap();
    let plan = provider
        .search_scales(&[&weight], &calibration, |c| {
            Ok(output_error(&weight, &c[0], &input))
        })
        .unwrap();
    assert_eq!(plan.candidate_losses.len(), 20);
    assert!(
        plan.loss < plan.candidate_losses[0],
        "{:?}",
        plan.candidate_losses
    );
    assert!(plan.ratio > 0. && plan.ratio < 1.);
    let minimum = plan
        .candidate_losses
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    assert_eq!(plan.ratio, minimum.0 as f64 / 20.);
    assert_eq!(plan.loss, *minimum.1);
    assert!(provider.quantize(&weight).is_err());
    let scaled = plan.input_weight(&weight).unwrap();
    let packed = provider.quantize_transformed(&scaled).unwrap();
    let effective =
        transforms::undo_input_weight(&provider.dequantize(&packed).unwrap(), plan.scales())
            .unwrap();
    assert_eq!(output_error(&weight, &effective, &input), plan.loss);
}

#[test]
fn paired_norm_and_swiglu_up_transforms_preserve_unquantized_functions() {
    let plan = AwqScalePlan {
        scales: (0..64).map(|i| 0.25 + (i % 7) as f32).collect(),
        ratio: 0.5,
        loss: 0.,
        candidate_losses: Vec::new(),
    };
    let norm =
        Tensor::from_f32(vec![64], (0..64).map(|i| 0.7 + i as f32 * 0.003).collect()).unwrap();
    let weight = weights();
    let scaled = plan.input_weight(&weight).unwrap();
    let normalized = plan.normalization(&norm).unwrap();
    for row in activations().as_f32().unwrap().as_chunks::<64>().0 {
        for o in 0..3 {
            let original: f64 = (0..64)
                .map(|k| {
                    f64::from(row[k])
                        * f64::from(norm.as_f32().unwrap()[k])
                        * f64::from(weight.as_f32().unwrap()[k * 3 + o])
                })
                .sum();
            let actual: f64 = (0..64)
                .map(|k| {
                    f64::from(row[k])
                        * f64::from(normalized.as_f32().unwrap()[k])
                        * f64::from(scaled.as_f32().unwrap()[k * 3 + o])
                })
                .sum();
            assert!((actual - original).abs() < 2e-6 * (1. + original.abs()));
        }
    }
    let up = Tensor::from_f32(
        vec![32, 64],
        (0..2048).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect(),
    )
    .unwrap();
    let changed_up = plan.output_weight(&up).unwrap();
    for c in 0..64 {
        let original: f64 = (0..32)
            .map(|k| up.as_f32().unwrap()[k * 64 + c] as f64 * (k as f64 * 0.01 - 0.1))
            .sum();
        let changed: f64 = (0..32)
            .map(|k| changed_up.as_f32().unwrap()[k * 64 + c] as f64 * (k as f64 * 0.01 - 0.1))
            .sum();
        // The unchanged SiLU gate multiplies both paths by the same factor.
        assert!((original - changed * f64::from(plan.scales[c])).abs() < 1e-7);
    }
}

#[test]
fn codes_are_nearest_on_the_final_f16_grid_for_both_native_encodings() {
    let weight = weights();
    let source = weight.as_f32().unwrap();
    for bits in [4, 8] {
        let provider = AwqProvider::new(AwqOptions {
            bits,
            group_size: 64,
            ..Default::default()
        })
        .unwrap();
        let packed = provider.quantize_transformed(&weight).unwrap();
        let decoded = provider.dequantize(&packed).unwrap();
        for o in 0..3 {
            for i in 0..64 {
                let block = o * 2 + i / 32;
                let offset = i % 32;
                let (scale, code) = match &packed {
                    QuantizedData::Q4_0(t) => (
                        t.blocks()[block].scale,
                        i32::from(t.blocks()[block].values[offset]) - 8,
                    ),
                    QuantizedData::Q8_0(t) => (
                        t.blocks()[block].scale,
                        i32::from(t.blocks()[block].values[offset]),
                    ),
                    _ => unreachable!(),
                };
                assert_eq!(f16::from_f32(scale).to_f32(), scale);
                assert_eq!(decoded.as_f32().unwrap()[i * 3 + o], scale * code as f32);
                let actual =
                    (f64::from(source[i * 3 + o]) - f64::from(scale) * f64::from(code)).abs();
                for alternative in -(1_i32 << (bits - 1))..(1_i32 << (bits - 1)) {
                    let error = (f64::from(source[i * 3 + o])
                        - f64::from(scale) * f64::from(alternative))
                    .abs();
                    assert!(actual <= error + 1e-14);
                }
            }
        }
    }
}

#[test]
fn statistics_count_actual_rows_reject_bad_batches_and_fail_before_evaluation() {
    let input = activations();
    let values = input.as_f32().unwrap();
    let mut whole = AwqCalibration::new(64, 512).unwrap();
    whole.observe_rows(&input).unwrap();
    let mut split = AwqCalibration::new(64, 512).unwrap();
    for rows in values.as_chunks::<{ 24 * 64 }>().0 {
        split
            .observe_rows(&Tensor::from_f32(vec![24, 64], rows.to_vec()).unwrap())
            .unwrap();
    }
    assert_eq!(whole, split);
    assert_eq!(whole.rows(), 48);
    let before = whole.clone();
    let invalid = Tensor::from_f32(vec![1, 64], vec![f32::NAN; 64]).unwrap();
    assert!(whole.observe_rows(&invalid).is_err());
    assert_eq!(whole, before);
    assert!(AwqCalibration::new(64, 511).is_err());
    let provider = AwqProvider::new(AwqOptions {
        maximum_workspace_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    let mut evaluated = false;
    assert!(provider
        .search_scales(&[&weights()], &whole, |_| {
            evaluated = true;
            Ok(0.)
        })
        .is_err());
    assert!(!evaluated);
    assert!(AwqProvider::default()
        .search_scales(
            &[&weights()],
            &AwqCalibration::new(64, 512).unwrap(),
            |_| Ok(0.)
        )
        .is_err());
    assert!(AwqProvider::default()
        .search_scales(&[&weights()], &whole, |_| Ok(f64::NAN))
        .is_err());
}
