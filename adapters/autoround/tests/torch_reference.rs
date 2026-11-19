use quantinfer_adapter_autoround::{AutoRoundCalibration, AutoRoundOptions, AutoRoundProvider};
use quantinfer_core::Tensor;
use quantinfer_quant::{QuantizationProvider, QuantizedData};
use serde_json::Value;

fn fixture() -> (Tensor, Tensor) {
    let weights = (0..96i32 * 3)
        .map(|i| {
            if i / 3 % 32 == 31 {
                [3.25, -2.5, 4.75][i as usize % 3]
            } else {
                (i * 17 % 127 - 63) as f32 / 256.
            }
        })
        .collect();
    let inputs = (0..23i32 * 96)
        .map(|i| {
            let channel = i % 96;
            let value = (i * 29 % 127 - 63) as f32 / 64. * ((channel % 13 + 1) as f32 / 4.);
            if channel % 32 == 31 {
                value / 128.
            } else {
                value
            }
        })
        .collect();
    (
        Tensor::from_f32(vec![96, 3], weights).unwrap(),
        Tensor::from_f32(vec![23, 96], inputs).unwrap(),
    )
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[test]
fn sign_round_matches_autograd_and_cpu_thread_counts() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/autoround-signed-profile.json"
    ))
    .unwrap();
    let (weights, inputs) = fixture();
    let mut calibration = AutoRoundCalibration::new(96, 23 * 96 * 4).unwrap();
    calibration.observe_rows(&inputs).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let options = AutoRoundOptions {
            bits: case["bits"].as_u64().unwrap() as usize,
            group_size: case["group_size"].as_u64().unwrap() as usize,
            iterations: case["iterations"].as_u64().unwrap() as usize,
            threads: 1,
            ..AutoRoundOptions::default()
        };
        let provider = AutoRoundProvider::new(options).unwrap();
        let result = provider.optimize(&weights, &calibration).unwrap();
        let threaded = AutoRoundProvider::new(AutoRoundOptions {
            threads: 3,
            ..options
        })
        .unwrap()
        .optimize(&weights, &calibration)
        .unwrap();
        assert_eq!(result.weights, threaded.weights);
        assert_eq!(result.initial_mse, threaded.initial_mse);
        assert_eq!(result.best_mse, threaded.best_mse);
        assert_eq!(result.improved_outputs, 3);
        assert!(result.best_mse < result.initial_mse);
        let (steps, codes): (Vec<_>, Vec<f64>) = match &result.weights {
            QuantizedData::Q4_0(t) => (
                t.blocks().iter().map(|b| f64::from(b.scale)).collect(),
                t.blocks()
                    .iter()
                    .flat_map(|b| b.values.map(|v| f64::from(v) - 8.))
                    .collect(),
            ),
            QuantizedData::Q8_0(t) => (
                t.blocks().iter().map(|b| f64::from(b.scale)).collect(),
                t.blocks()
                    .iter()
                    .flat_map(|b| b.values.map(f64::from))
                    .collect(),
            ),
            _ => panic!("unexpected native bit width"),
        };
        assert_eq!(steps, numbers(&case["scales_output_rows"]));
        assert_eq!(codes, numbers(&case["codes_output_rows"]));
        assert!((case["initial_mse"].as_f64().unwrap() - result.initial_mse).abs() < 1e-10);
        assert!((case["best_mse"].as_f64().unwrap() - result.best_mse).abs() < 1e-10);
        let decoded = provider.dequantize(&result.weights).unwrap();
        let error: Vec<_> = decoded
            .as_f32()
            .unwrap()
            .iter()
            .zip(weights.as_f32().unwrap())
            .map(|(&a, &b)| f64::from(a) - f64::from(b))
            .collect();
        let measured = inputs
            .as_f32()
            .unwrap()
            .as_chunks::<96>()
            .0
            .iter()
            .flat_map(|row| {
                let error = &error;
                (0..3).map(move |o| {
                    row.iter()
                        .enumerate()
                        .map(|(i, &x)| f64::from(x) * error[i * 3 + o])
                        .sum::<f64>()
                        .powi(2)
                })
            })
            .sum::<f64>()
            / (23. * 3.);
        assert!((measured - result.best_mse).abs() < 1e-10);
    }
}

#[test]
fn requires_finite_calibration_rows_and_enforces_budgets() {
    let (weights, inputs) = fixture();
    let mut calibration = AutoRoundCalibration::new(96, 23 * 96 * 4).unwrap();
    let provider = AutoRoundProvider::new(AutoRoundOptions {
        threads: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(provider.optimize(&weights, &calibration).is_err());
    assert!(provider.quantize(&weights).is_err());
    calibration.observe_rows(&inputs).unwrap();
    assert!(calibration.observe_rows(&inputs).is_err());
    assert_eq!(calibration.rows(), 23);
    let small = AutoRoundProvider::new(AutoRoundOptions {
        threads: 1,
        maximum_workspace_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    assert!(small
        .optimize(&weights, &calibration)
        .unwrap_err()
        .to_string()
        .contains("workspace budget"));
}
