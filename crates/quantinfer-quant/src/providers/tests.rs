use super::{FloatProvider, QuantizationProvider, QuantizationStats};
use quantinfer_core::Tensor;

fn tensor(values: [f32; 2]) -> Tensor {
    Tensor::from_f32(vec![2], values.to_vec()).unwrap()
}

#[test]
fn finite_weights_have_finite_mean_absolute_value() {
    let stats = QuantizationStats::inspect(&tensor([3.0e38, -3.0e38])).unwrap();
    assert_eq!(stats.mean_abs, 3.0e38);
    assert!(stats.mean_abs.is_finite());
}

#[test]
fn relative_l2_is_scale_invariant_for_large_and_small_weights() {
    let zero = FloatProvider.quantize(&tensor([0.0; 2])).unwrap();
    for scale in [1.0e38, 1.0, 1.0e-30, f32::from_bits(1)] {
        let report = FloatProvider
            .error_report(&tensor([scale, -scale]), &zero)
            .unwrap();
        assert_eq!(report.max_absolute_error, scale);
        assert_eq!(report.mean_absolute_error, scale);
        assert_eq!(report.relative_l2_error, 1.0);
    }
}

#[test]
fn undefined_or_unrepresentable_metrics_are_errors() {
    let zero = tensor([0.0; 2]);
    let packed_zero = FloatProvider.quantize(&zero).unwrap();
    assert_eq!(
        FloatProvider
            .error_report(&zero, &packed_zero)
            .unwrap()
            .relative_l2_error,
        0.0
    );
    let nonzero = FloatProvider.quantize(&tensor([1.0, -1.0])).unwrap();
    assert!(FloatProvider.error_report(&zero, &nonzero).is_err());
    let opposite = FloatProvider.quantize(&tensor([-f32::MAX; 2])).unwrap();
    assert!(FloatProvider
        .error_report(&tensor([f32::MAX; 2]), &opposite)
        .is_err());
}
