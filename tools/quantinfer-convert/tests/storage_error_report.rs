//! Independent format round trips for the quantization storage-precision API.
use quantinfer_core::{Q8_0Block, QuantizedTensor, Tensor};
use quantinfer_format::{GgufWriter, PackedTensorReader};
use quantinfer_quant::{
    measure_linear_precision_candidates, FloatProvider, NativeQ4KProvider, NativeQ4_0Provider,
    NativeQ5KProvider, NativeQ6KProvider, NativeQ8_0Provider, QuantizationProvider, QuantizedData,
};

fn encode(data: &QuantizedData) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    match data {
        QuantizedData::Float(tensor) => writer
            .add_f32_tensor(
                "weight",
                tensor.shape().dims().to_vec(),
                tensor.as_f32().unwrap(),
            )
            .unwrap(),
        QuantizedData::Q8_0(tensor) => writer.add_q8_0_tensor("weight", tensor),
        QuantizedData::Q4_0(tensor) => writer.add_q4_0_tensor("weight", tensor),
        QuantizedData::Q4_K(tensor) => writer.add_q4_k_tensor("weight", tensor),
        QuantizedData::Q5_K(tensor) => writer.add_q5_k_tensor("weight", tensor),
        QuantizedData::Q6_K(tensor) => writer.add_q6_k_tensor("weight", tensor),
    }
    writer.finish().unwrap()
}

fn round_trip(data: &QuantizedData) -> QuantizedData {
    let reader = PackedTensorReader::open(encode(data)).unwrap();
    match data {
        QuantizedData::Float(_) => QuantizedData::Float(reader.tensor_f32("weight").unwrap()),
        QuantizedData::Q8_0(_) => QuantizedData::Q8_0(reader.tensor_q8_0("weight").unwrap()),
        QuantizedData::Q4_0(_) => QuantizedData::Q4_0(reader.tensor_q4_0("weight").unwrap()),
        QuantizedData::Q4_K(_) => QuantizedData::Q4_K(reader.tensor_q4_k("weight").unwrap()),
        QuantizedData::Q5_K(_) => QuantizedData::Q5_K(reader.tensor_q5_k("weight").unwrap()),
        QuantizedData::Q6_K(_) => QuantizedData::Q6_K(reader.tensor_q6_k("weight").unwrap()),
    }
}

#[test]
fn storage_reports_match_all_fixed_encodings_re_read_from_the_format() {
    let providers: [&dyn QuantizationProvider; 6] = [
        &FloatProvider,
        &NativeQ8_0Provider,
        &NativeQ4_0Provider,
        &NativeQ4KProvider,
        &NativeQ5KProvider,
        &NativeQ6KProvider,
    ];
    for values in [
        (0..256)
            .map(|i| ((i * 37 % 71) as f32 - 35.0) * 0.0137)
            .collect(),
        vec![1.0e-8; 256],
        vec![5.0e-7; 256],
    ] {
        let input = Tensor::from_f32(vec![8, 32], values).unwrap();
        for provider in providers {
            let data = provider.quantize(&input).unwrap();
            let original = data.clone();
            let stored = data.with_storage_precision().unwrap();
            let decoded = round_trip(&data);
            let measurements =
                measure_linear_precision_candidates("weight", &input, &[provider.scheme()], None)
                    .unwrap();
            let choice = &measurements.group.candidates[0];
            let file = PackedTensorReader::open(encode(&data)).unwrap();
            assert_eq!(
                choice.encoded_bytes,
                file.tensor_info("weight").unwrap().size as u64
            );
            let restored = provider.dequantize(&decoded).unwrap();
            let expected = input
                .as_f32()
                .unwrap()
                .iter()
                .zip(restored.as_f32().unwrap())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>();
            assert_eq!(choice.error, expected);
            assert_eq!(stored, decoded);
            assert_eq!(encode(&data), encode(&stored));
            assert_eq!(
                provider.error_report_for_storage(&input, &data).unwrap(),
                provider.error_report(&input, &decoded).unwrap()
            );
            assert_eq!(data, original);
        }
    }
}

#[test]
fn reports_f16_underflow_instead_of_reusing_the_in_memory_zero_error() {
    let input = Tensor::from_f32(vec![32], vec![1.0e-8; 32]).unwrap();
    let providers: [&dyn QuantizationProvider; 2] = [&NativeQ8_0Provider, &NativeQ4_0Provider];
    for provider in providers {
        let data = provider.quantize(&input).unwrap();
        assert_eq!(
            provider
                .error_report(&input, &data)
                .unwrap()
                .relative_l2_error,
            0.0
        );
        assert_eq!(
            provider
                .error_report_for_storage(&input, &data)
                .unwrap()
                .relative_l2_error,
            1.0
        );
    }
}

#[test]
fn rejects_unrepresentable_storage_without_restricting_core_scales() {
    let data = QuantizedData::Q8_0(
        QuantizedTensor::from_blocks(
            vec![32],
            vec![Q8_0Block {
                scale: 1.0e10,
                values: [0; 32],
            }],
        )
        .unwrap(),
    );
    assert!(data.with_storage_precision().is_err());
}
