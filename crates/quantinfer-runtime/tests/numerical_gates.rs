use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{quantize_q4_0, quantize_q8_0};
use quantinfer_runtime::{select_token, GenerationConfig};

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|left, right| values[*right].total_cmp(&values[*left]));
    indices.truncate(k.min(indices.len()));
    indices
}

#[test]
fn quantized_weight_error_and_logits_gates_are_explicit() {
    let input = Tensor::from_f32(
        vec![2, 32],
        (0..64)
            .map(|index| ((index as f32 * 0.17).sin()) * 0.75)
            .collect(),
    )
    .unwrap();
    let weight = Tensor::from_f32(
        vec![32, 8],
        (0..256)
            .map(|index| ((index as f32 * 0.11).cos()) * 0.4)
            .collect(),
    )
    .unwrap();
    let q8 = quantize_q8_0(&weight).unwrap();
    let q4 = quantize_q4_0(&weight).unwrap();
    let q8_restored = quantinfer_quant::dequantize_q8_0(&q8).unwrap();
    let q4_restored = quantinfer_quant::dequantize_q4_0(&q4).unwrap();
    let reference = weight.as_f32().unwrap();
    let q8_error = q8_restored
        .as_f32()
        .unwrap()
        .iter()
        .zip(reference)
        .map(|(restored, expected)| (restored - expected).abs())
        .fold(0.0, f32::max);
    let q4_error = q4_restored
        .as_f32()
        .unwrap()
        .iter()
        .zip(reference)
        .map(|(restored, expected)| (restored - expected).abs())
        .fold(0.0, f32::max);
    assert!(
        q8_error < 0.01,
        "Q8_0 weight error {q8_error} exceeded gate"
    );
    assert!(
        q4_error < 0.08,
        "Q4_0 weight error {q4_error} exceeded gate"
    );

    let backend = CpuScalar::new();
    let f32_logits = backend.matmul(&input, &weight).unwrap();
    let q8_logits = backend.matmul_q8_0(&input, &q8).unwrap();
    let q4_logits = backend.matmul_q4_0(&input, &q4).unwrap();
    let q8_values = q8_logits.as_f32().unwrap();
    let q4_values = q4_logits.as_f32().unwrap();
    let f32_values = f32_logits.as_f32().unwrap();
    let q8_max = f32_values
        .iter()
        .zip(q8_values)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f32::max);
    let q8_mse = f32_values
        .iter()
        .zip(q8_values)
        .map(|(expected, actual)| (expected - actual).powi(2))
        .sum::<f32>()
        / f32_values.len() as f32;
    assert!(
        q8_max < 0.02,
        "Q8_0 logits max error {q8_max} exceeded gate"
    );
    assert!(q8_mse < 0.0001, "Q8_0 logits MSE {q8_mse} exceeded gate");
    let f32_top = top_k(f32_values, 4);
    let q8_top = top_k(q8_values, 4);
    let q4_top = top_k(q4_values, 4);
    assert!(f32_top.iter().filter(|id| q8_top.contains(id)).count() >= 3);
    assert!(f32_top.iter().filter(|id| q4_top.contains(id)).count() >= 2);
}

#[test]
fn generation_selection_is_deterministic_and_honors_eos() {
    let logits = Tensor::from_f32(vec![4], vec![0.2, 0.8, 0.1, 0.7]).unwrap();
    let config = GenerationConfig {
        max_new_tokens: 4,
        eos_token_id: Some(1),
        ..GenerationConfig::default()
    };
    let first = select_token(&logits, &config).unwrap();
    let second = select_token(&logits, &config).unwrap();
    assert_eq!(first, second);
    assert_eq!(first, config.eos_token_id.unwrap());
}
