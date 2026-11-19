use quantinfer_adapter_awq::{AwqCalibration, AwqOptions, AwqProvider};
use quantinfer_core::Tensor;
use quantinfer_quant::QuantizedData;
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
fn native_awq_matches_independent_torch_scale_clipping_and_code_vectors() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/awq-signed-profile.json"
    ))
    .unwrap();
    let (weights, inputs) = fixture();
    let x = inputs.as_f32().unwrap();
    let w = weights.as_f32().unwrap();
    let mut stats = AwqCalibration::new(96, 96 * 8).unwrap();
    stats.observe_rows(&inputs).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let bits = case["bits"].as_u64().unwrap() as usize;
        let group_size = case["group_size"].as_u64().unwrap() as usize;
        let provider = AwqProvider::new(AwqOptions {
            bits,
            group_size,
            ..AwqOptions::default()
        })
        .unwrap();
        let scales = provider
            .search_scales(&[&weights], &stats, |effective| {
                let candidate = effective[0].as_f32()?;
                let mut squared = 0.;
                for row in x.as_chunks::<96>().0 {
                    for output in 0..3 {
                        let delta = row
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                f64::from(v)
                                    * (f64::from(candidate[i * 3 + output])
                                        - f64::from(w[i * 3 + output]))
                            })
                            .sum::<f64>();
                        squared += delta * delta;
                    }
                }
                Ok(squared / (23. * 3.))
            })
            .unwrap();
        assert_eq!(scales.ratio, case["best_ratio"].as_f64().unwrap());
        let expected_losses = numbers(&case["candidate_losses"]);
        assert_eq!(scales.candidate_losses.len(), expected_losses.len());
        for (actual, expected) in scales.candidate_losses.iter().zip(expected_losses) {
            assert!((actual - expected).abs() < 1e-10);
        }
        let expected_scales: Vec<_> = numbers(&case["channel_scales"])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        assert_eq!(scales.scales(), expected_scales);
        let scaled = scales.input_weight(&weights).unwrap();
        let scaled_inputs = Tensor::from_f32(
            inputs.shape().clone(),
            x.iter()
                .enumerate()
                .map(|(i, &v)| v / scales.scales()[i % 96])
                .collect(),
        )
        .unwrap();
        let clipping = provider.search_clipping(&scaled, &scaled_inputs).unwrap();
        assert_eq!(
            clipping.bounds,
            numbers(&case["clipping_bounds"])
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>()
        );
        for (key, actual) in [
            ("group_loss_before", clipping.group_loss_before),
            ("group_loss_after", clipping.group_loss_after),
        ] {
            assert!((case[key].as_f64().unwrap() - actual).abs() < 1e-10);
        }
        let (steps, codes): (Vec<_>, Vec<f64>) =
            match provider.quantize_transformed(&clipping.weights).unwrap() {
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
    }
}
