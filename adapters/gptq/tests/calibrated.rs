use quantinfer_adapter_gptq::{GptqProvider, GptqQuantizationOptions, HessianCalibration};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{Shape, Tensor};
use quantinfer_format::{qif, GgufReader, GgufWriter, ModelManifest, TensorManifest};
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{CalibratedQuantizationProvider, QuantizationProvider, QuantizedData};
use serde_json::Value;
use std::collections::BTreeMap;

fn tensor(document: &Value, shape: &str, values: &str) -> Tensor {
    let dimensions: Vec<usize> = document[shape]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    Tensor::from_f32(
        dimensions,
        document[values]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
    .unwrap()
}

#[test]
fn matches_pinned_upstream_feedback_then_publishes_and_executes_without_requantizing() {
    let document: Value = serde_json::from_str(include_str!(
        "../../../tests/golden/gptq-static-profile.json"
    ))
    .unwrap();
    let weights = tensor(&document, "weights_shape", "weights");
    let activations = tensor(&document, "activation_shape", "activations");
    let mut calibration = HessianCalibration::new(64, 64 * 64 * 8).unwrap();
    calibration.observe_rows(&activations).unwrap();
    for case in document["cases"].as_array().unwrap() {
        let options = GptqQuantizationOptions {
            bits: case["bits"].as_u64().unwrap() as usize,
            group_size: case["group_size"].as_u64().unwrap() as usize,
            activation_order: case["activation_order"].as_bool().unwrap(),
            ..Default::default()
        };
        let provider = GptqProvider::new(options).unwrap();
        let quantized = provider
            .quantize_calibrated(&weights, &calibration)
            .unwrap();
        let restored = provider.dequantize(&quantized).unwrap();
        let (scales, codes): (Vec<f32>, Vec<u8>) = match &quantized {
            QuantizedData::Q4_0(t) => (
                t.blocks().iter().map(|b| b.scale).collect(),
                t.blocks().iter().flat_map(|b| b.values).collect(),
            ),
            QuantizedData::Q8_0(t) => (
                t.blocks().iter().map(|b| b.scale).collect(),
                t.blocks()
                    .iter()
                    .flat_map(|b| b.values.map(|x| (i16::from(x) + 128) as u8))
                    .collect(),
            ),
            _ => unreachable!(),
        };
        let expected_scales: Vec<f32> = case["scales_output_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let expected_codes: Vec<u8> = case["codes_output_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(scales, expected_scales, "{options:?}");
        assert_eq!(codes, expected_codes, "{options:?}");

        let expected = CpuScalar::new().matmul(&activations, &restored).unwrap();
        let original = CpuScalar::new().matmul(&activations, &weights).unwrap();
        let mse = expected
            .as_f32()
            .unwrap()
            .iter()
            .zip(original.as_f32().unwrap())
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / expected.as_f32().unwrap().len() as f64;
        assert!(
            (mse - case["projection_mse"].as_f64().unwrap()).abs() < 1e-5,
            "{mse}: {options:?}"
        );
        assert!(
            mse < case["rtn_projection_mse"].as_f64().unwrap(),
            "feedback must improve this fixed reference case: {options:?}"
        );

        let mut packed = GgufWriter::new();
        match &quantized {
            QuantizedData::Q4_0(t) => packed.add_q4_0_tensor("projection", t),
            QuantizedData::Q8_0(t) => packed.add_q8_0_tensor("projection", t),
            _ => unreachable!(),
        }
        let packed = GgufReader::open(packed.finish().unwrap()).unwrap();
        let mut manifest = ModelManifest::new("calibrated-gptq-reference", "test");
        manifest.quantization = provider.scheme();
        manifest.tensors.push(TensorManifest {
            name: "projection".into(),
            descriptor: provider
                .storage_descriptor(Shape::from(vec![64, 3]))
                .unwrap(),
            checksum_sha256: None,
        });
        for codec in [qif::Codec::None, qif::Codec::Zstd, qif::Codec::Lz4] {
            let mut bytes = Vec::new();
            qif::write_qif_with_codec(
                manifest.clone(),
                &BTreeMap::new(),
                |_| packed.tensor_bytes("projection"),
                &mut bytes,
                codec,
            )
            .unwrap();
            let native = qif::Reader::open(&bytes).unwrap();
            native.verify().unwrap();
            assert_eq!(
                native.tensor_data("projection").unwrap().as_slice(),
                packed.tensor_bytes("projection").unwrap()
            );
            let loaded = GgufReader::open_artifact(bytes).unwrap();
            let actual = if options.bits == 4 {
                let reloaded = loaded.tensor_q4_0("projection").unwrap();
                if let QuantizedData::Q4_0(original) = &quantized {
                    assert_eq!(original.blocks(), reloaded.blocks());
                }
                CpuScalar::new()
                    .matmul_q4_0_output_rows(&activations, &reloaded)
                    .unwrap()
            } else {
                let reloaded = loaded.tensor_q8_0("projection").unwrap();
                if let QuantizedData::Q8_0(original) = &quantized {
                    assert_eq!(original.blocks(), reloaded.blocks());
                }
                CpuScalar::new()
                    .matmul_q8_0_output_rows(&activations, &reloaded)
                    .unwrap()
            };
            let error = actual
                .as_f32()
                .unwrap()
                .iter()
                .zip(expected.as_f32().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            assert!(error < 1e-4, "{error}: {options:?}/{codec:?}");
        }
    }
}
