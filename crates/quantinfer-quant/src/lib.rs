//! Weight-only quantization primitives.

#![forbid(unsafe_code)]

pub type Result<T> = std::result::Result<T, QuantizationError>;

mod imatrix;
mod providers;
mod schemes;
mod types;

pub use imatrix::ImatrixQ4_0Provider;
pub use providers::{
    FloatProvider, GgmlQ8_0Provider, NativeQ4KProvider, NativeQ4_0Provider, NativeQ5KProvider,
    NativeQ6KProvider, NativeQ8_0Provider, QuantizationErrorReport, QuantizationProvider,
    QuantizationStats, QuantizedData,
};
pub use schemes::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0,
};
pub use types::{CalibrationDataset, ImatrixStatistics, MixedPrecisionPlan, QuantizationError};

#[cfg(test)]
mod tests {
    use super::{
        dequantize_q8_0, quantize_q8_0, CalibrationDataset, ImatrixStatistics, MixedPrecisionPlan,
        NativeQ4KProvider, NativeQ4_0Provider, NativeQ6KProvider, QuantizationProvider,
    };
    use quantinfer_core::{QuantizationScheme, Tensor};

    #[test]
    fn quantizes_and_dequantizes_a_block() {
        let input =
            Tensor::from_f32(vec![32], (0..32).map(|value| value as f32).collect()).unwrap();
        let quantized = quantize_q8_0(&input).unwrap();
        assert_eq!(quantized.blocks().len(), 1);
        let restored = dequantize_q8_0(&quantized).unwrap();
        let values = restored.as_f32().unwrap();
        assert!((values[31] - 31.0).abs() < 0.3);
    }

    #[test]
    fn rejects_non_block_aligned_input() {
        let input = Tensor::from_f32(vec![31], vec![1.0; 31]).unwrap();
        assert!(quantize_q8_0(&input).is_err());
    }

    #[test]
    fn rejects_non_finite_quantization_input() {
        let input = Tensor::from_f32(vec![32], vec![f32::NAN; 32]).unwrap();
        assert!(quantize_q8_0(&input).is_err());
        assert!(NativeQ4_0Provider.quantize(&input).is_err());
    }

    #[test]
    fn quantizes_q4_0_and_reports_error() {
        let input = Tensor::from_f32(
            vec![32],
            (0..32).map(|value| value as f32 / 4.0 - 4.0).collect(),
        )
        .unwrap();
        let provider = NativeQ4_0Provider;
        let quantized = provider.quantize(&input).unwrap();
        let restored = provider.dequantize(&quantized).unwrap();
        assert!(
            provider
                .error_report(&input, &quantized)
                .unwrap()
                .max_absolute_error
                < 0.4
        );
        assert_eq!(restored.shape(), input.shape());
        assert_eq!(provider.export_layout(), "q4_0");
    }

    #[test]
    fn weights_imatrix_by_token_rows_across_different_prompt_lengths() {
        let mut stats = ImatrixStatistics::new(2).unwrap();
        let first = Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let second = Tensor::from_f32(vec![1, 2], vec![5.0, 6.0]).unwrap();
        stats.observe_rows(&first).unwrap();
        stats.observe_rows(&second).unwrap();
        assert_eq!(stats.count(), 3);
        assert_eq!(stats.mean_abs(), [3.0, 4.0]);
        assert_eq!(stats.mean_square(), [35.0 / 3.0, 56.0 / 3.0]);
        let invalid = Tensor::from_f32(vec![1, 2], vec![0.0, f32::NAN]).unwrap();
        let previous = stats.clone();
        assert!(stats.observe_rows(&invalid).is_err());
        assert_eq!(stats, previous);
    }

    #[test]
    fn builds_calibration_statistics_and_mixed_precision_assignments() {
        let dataset = CalibrationDataset::new(vec![
            Tensor::from_f32(vec![2], vec![-1.0, 2.0]).unwrap(),
            Tensor::from_f32(vec![2], vec![3.0, -4.0]).unwrap(),
        ])
        .unwrap();
        let statistics = ImatrixStatistics::from_dataset(&dataset).unwrap();
        assert_eq!(statistics.count(), 2);
        assert_eq!(statistics.mean_abs(), vec![2.0, 3.0]);
        let mut plan = MixedPrecisionPlan::default();
        plan.assign("blk.0.attn_q.weight", QuantizationScheme::Q8_0);
        assert_eq!(
            plan.scheme_for("blk.0.attn_q.weight"),
            Some(QuantizationScheme::Q8_0)
        );
    }

    #[test]
    fn quantizes_and_dequantizes_standard_q4_k_blocks() {
        let input = Tensor::from_f32(
            vec![256],
            (0..256).map(|index| index as f32 / 128.0 - 1.0).collect(),
        )
        .unwrap();
        let provider = NativeQ4KProvider;
        let quantized = provider.quantize(&input).unwrap();
        let report = provider.error_report(&input, &quantized).unwrap();
        assert!(report.max_absolute_error < 0.2);
        assert_eq!(provider.export_layout(), "q4_k");
    }

    #[test]
    fn quantizes_and_dequantizes_standard_q6_k_blocks() {
        let input = Tensor::from_f32(
            vec![256],
            (0..256).map(|index| index as f32 / 128.0 - 1.0).collect(),
        )
        .unwrap();
        let provider = NativeQ6KProvider;
        let quantized = provider.quantize(&input).unwrap();
        let report = provider.error_report(&input, &quantized).unwrap();
        assert!(report.max_absolute_error < 0.2);
    }
}
