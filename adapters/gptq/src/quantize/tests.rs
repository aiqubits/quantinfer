use super::*;

#[test]
fn full_gram_preserves_correlations_and_is_independent_of_batch_boundaries() {
    let rows = Tensor::from_f32(vec![3, 2], vec![1., 2., 3., -4., -1., 5.]).unwrap();
    let mut all = HessianCalibration::new(2, 32).unwrap();
    all.observe_rows(&rows).unwrap();
    assert_eq!(all.rows(), 3);
    assert_eq!(all.value(0, 0), 22. / 3.);
    assert_eq!(all.value(1, 1), 30.);
    assert_eq!(all.value(0, 1), -10.);
    let mut split = HessianCalibration::new(2, 32).unwrap();
    for row in rows.as_f32().unwrap().as_chunks::<2>().0 {
        split
            .observe_rows(&Tensor::from_f32(vec![1, 2], row.to_vec()).unwrap())
            .unwrap();
    }
    assert_eq!(all, split);
    let previous = all.clone();
    assert!(all
        .observe_rows(&Tensor::from_f32(vec![1, 2], vec![f32::NAN, 1.]).unwrap())
        .is_err());
    assert_eq!(previous, all);
    assert!(HessianCalibration::new(2, 31).is_err());
    assert!(HessianCalibration::new(usize::MAX, usize::MAX).is_err());
}

#[test]
fn inverse_factor_satisfies_full_spd_matrix_identity() {
    let matrix = [4., 1., 2., 1., 3., -1., 2., -1., 6.];
    let mut lower = matrix.to_vec();
    for i in 0..3 {
        for j in i + 1..3 {
            lower[i * 3 + j] = 0.;
        }
    }
    let upper = linalg::inverse_factor(lower, 3).unwrap();
    let mut inverse = [0.; 9];
    for i in 0..3 {
        for j in 0..3 {
            inverse[i * 3 + j] = (0..3).map(|k| upper[k * 3 + i] * upper[k * 3 + j]).sum();
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            let actual: f64 = (0..3).map(|k| matrix[i * 3 + k] * inverse[k * 3 + j]).sum();
            assert!(
                (actual - f64::from(i == j)).abs() < 1e-12,
                "{i}/{j}: {actual}"
            );
        }
    }
    assert!(linalg::inverse_factor(vec![1., 0., 2., 1.], 2).is_err());
}

#[test]
fn missing_invalid_or_over_budget_calibration_cannot_fall_back_to_rtn() {
    let weights = Tensor::from_f32(vec![32, 1], vec![1.; 32]).unwrap();
    let provider = GptqProvider::default();
    assert!(provider
        .quantize(&weights)
        .unwrap_err()
        .to_string()
        .contains("calibration"));
    let mut stats = HessianCalibration::new(32, 32 * 32 * 8).unwrap();
    assert!(provider.quantize_calibrated(&weights, &stats).is_err());
    stats
        .observe_rows(&Tensor::from_f32(vec![1, 32], vec![0.; 32]).unwrap())
        .unwrap();
    assert!(provider.quantize_calibrated(&weights, &stats).is_err());
    stats
        .observe_rows(&Tensor::from_f32(vec![1, 32], vec![1.; 32]).unwrap())
        .unwrap();
    let bounded = GptqProvider::new(GptqQuantizationOptions {
        maximum_workspace_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(bounded
        .quantize_calibrated(&weights, &stats)
        .unwrap_err()
        .to_string()
        .contains("workspace"));
    assert!(provider
        .quantize_calibrated(
            &Tensor::from_f32(vec![32, 1], vec![f32::MAX; 32]).unwrap(),
            &stats
        )
        .is_err());
    let extreme = GptqProvider::new(GptqQuantizationOptions {
        damping: f64::MAX,
        ..Default::default()
    })
    .unwrap();
    let mut high = HessianCalibration::new(32, 32 * 32 * 8).unwrap();
    high.observe_rows(&Tensor::from_f32(vec![1, 32], vec![2.; 32]).unwrap())
        .unwrap();
    assert!(extreme
        .quantize_calibrated(&weights, &high)
        .unwrap_err()
        .to_string()
        .contains("damping"));
    let mut tiny = HessianCalibration::new(32, 32 * 32 * 8).unwrap();
    tiny.observe_rows(&Tensor::from_f32(vec![1, 32], vec![1e-30; 32]).unwrap())
        .unwrap();
    let underflow = GptqProvider::new(GptqQuantizationOptions {
        damping: f64::MIN_POSITIVE,
        ..Default::default()
    })
    .unwrap();
    assert!(underflow
        .quantize_calibrated(&weights, &tiny)
        .unwrap_err()
        .to_string()
        .contains("damping"));
    for options in [
        GptqQuantizationOptions {
            bits: 2,
            ..Default::default()
        },
        GptqQuantizationOptions {
            group_size: 0,
            ..Default::default()
        },
        GptqQuantizationOptions {
            group_size: 33,
            ..Default::default()
        },
        GptqQuantizationOptions {
            damping: f64::NAN,
            ..Default::default()
        },
        GptqQuantizationOptions {
            damping: 0.,
            ..Default::default()
        },
    ] {
        assert!(GptqProvider::new(options).is_err());
    }
}

#[test]
fn dead_channels_and_subnormal_scales_produce_finite_native_weights() {
    let mut activations = vec![1.; 32];
    activations[17] = 0.;
    let mut stats = HessianCalibration::new(32, 32 * 32 * 8).unwrap();
    stats
        .observe_rows(&Tensor::from_f32(vec![1, 32], activations).unwrap())
        .unwrap();
    for bits in [4, 8] {
        let provider = GptqProvider::new(GptqQuantizationOptions {
            bits,
            activation_order: true,
            ..Default::default()
        })
        .unwrap();
        let input = Tensor::from_f32(vec![32, 1], vec![1e-8; 32]).unwrap();
        let result = provider.quantize_calibrated(&input, &stats).unwrap();
        let restored = provider.dequantize(&result).unwrap();
        assert!(restored.as_f32().unwrap().iter().all(|x| x.is_finite()));
        assert_eq!(restored.as_f32().unwrap()[17], 0.);
        let scales: Vec<f32> = match result {
            QuantizedData::Q4_0(t) => t.blocks().iter().map(|b| b.scale).collect(),
            QuantizedData::Q8_0(t) => t.blocks().iter().map(|b| b.scale).collect(),
            _ => unreachable!(),
        };
        assert_eq!(scales, [half::f16::from_bits(1).to_f32()]);
    }
}

#[test]
fn signed_grids_use_ties_to_even_and_static_groups_keep_partial_tail_blocks() {
    for (channels, group_size) in [(32, 32), (96, 64)] {
        let mut identity = vec![0.; channels * channels];
        for i in 0..channels {
            identity[i * channels + i] = 1.;
        }
        let mut statistics = HessianCalibration::new(channels, channels * channels * 8).unwrap();
        statistics
            .observe_rows(&Tensor::from_f32(vec![channels, channels], identity).unwrap())
            .unwrap();
        for bits in [4, 8] {
            let maximum = ((1_u32 << bits) - 1) as f32;
            let mut source = vec![0.; channels];
            for start in (0..channels).step_by(group_size) {
                source[start] = maximum;
                source[start + 1] = -maximum;
                source[start + 2..start + 6].copy_from_slice(&[1., 3., -1., -3.]);
            }
            let provider = GptqProvider::new(GptqQuantizationOptions {
                bits,
                group_size,
                ..Default::default()
            })
            .unwrap();
            let result = provider
                .quantize_calibrated(
                    &Tensor::from_f32(vec![channels, 1], source).unwrap(),
                    &statistics,
                )
                .unwrap();
            let restored = provider.dequantize(&result).unwrap();
            for start in (0..channels).step_by(group_size) {
                assert_eq!(
                    &restored.as_f32().unwrap()[start + 2..start + 6],
                    &[0., 4., 0., -4.]
                );
            }
        }
    }
}
