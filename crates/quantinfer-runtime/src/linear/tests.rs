use super::*;
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{QuantizationScheme, TensorStorageDescriptor};
use quantinfer_format::{qif, GgufWriter, ModelManifest, TensorManifest};
use quantinfer_quant::{quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0};
use std::collections::BTreeMap;

fn physical_source(dtype: DType, shape: &[usize], values: &[f32]) -> GgufReader {
    let mut writer = GgufWriter::new();
    let tensor = Tensor::from_f32(shape.to_vec(), values.to_vec()).unwrap();
    match dtype {
        DType::F32 => writer
            .add_f32_tensor("weight", shape.to_vec(), values)
            .unwrap(),
        DType::F16 => writer
            .add_f16_tensor("weight", shape.to_vec(), values)
            .unwrap(),
        DType::BF16 => writer
            .add_bf16_tensor("weight", shape.to_vec(), values)
            .unwrap(),
        DType::Q8_0 => writer.add_q8_0_tensor("weight", &quantize_q8_0(&tensor).unwrap()),
        DType::Q4_0 => writer.add_q4_0_tensor("weight", &quantize_q4_0(&tensor).unwrap()),
        DType::Q4_K => writer.add_q4_k_tensor("weight", &quantize_q4_k(&tensor).unwrap()),
        DType::Q5_K => writer.add_q5_k_tensor("weight", &quantize_q5_k(&tensor).unwrap()),
        DType::Q6_K => writer.add_q6_k_tensor("weight", &quantize_q6_k(&tensor).unwrap()),
    }
    GgufReader::open(writer.finish().unwrap()).unwrap()
}

fn fixture(
    dtype: DType,
    shape: &[usize],
    values: &[f32],
    ggml: bool,
    output_rows: bool,
) -> (GgufReader, GgufReader) {
    let source = physical_source(dtype, shape, values);
    let (scheme, block) = match dtype {
        DType::F32 | DType::F16 | DType::BF16 => (QuantizationScheme::None, 0),
        DType::Q8_0 => (QuantizationScheme::Q8_0, 32),
        DType::Q4_0 => (QuantizationScheme::Q4_0, 32),
        DType::Q4_K => (QuantizationScheme::Q4_K, 256),
        DType::Q5_K => (QuantizationScheme::Q5_K, 256),
        DType::Q6_K => (QuantizationScheme::Q6_K, 256),
    };
    let logical: Shape = if ggml {
        shape.iter().rev().copied().collect::<Vec<_>>()
    } else {
        shape.to_vec()
    }
    .into();
    let mut descriptor = if block == 0 {
        TensorStorageDescriptor::floating(dtype, logical).unwrap()
    } else {
        TensorStorageDescriptor::quantized(dtype, scheme, logical, block).unwrap()
    };
    if output_rows {
        descriptor.physical_layout = quantinfer_format::GGML_OUTPUT_ROWS_LAYOUT.into();
    }
    let mut manifest = ModelManifest::new("matrix-layout", "llama");
    manifest.quantization = scheme;
    manifest.tensors.push(TensorManifest {
        name: "weight".into(),
        descriptor,
        checksum_sha256: None,
    });
    let metadata = BTreeMap::from([(
        "quantinfer.qif.ggml_dimension_order".into(),
        qif::MetadataValue::Bool(ggml),
    )]);
    let raw = source.tensor_bytes("weight").unwrap();
    let mut bytes = Vec::new();
    qif::write_qif(manifest, &metadata, |_| Ok(raw), &mut bytes).unwrap();
    qif::Reader::open(&bytes).unwrap().verify().unwrap();
    (GgufReader::open_artifact(bytes).unwrap(), source)
}

#[test]
fn non_square_core_matrices_follow_each_layout_and_dimension_order() {
    let (inner, columns, rows) = (256, 32, 3);
    let input: Vec<_> = (0..rows * inner)
        .map(|i| (i * 19 % 97) as f32 / 97.0 - 0.5)
        .collect();
    let input_tensor = Tensor::from_f32(vec![rows, inner], input.clone()).unwrap();
    let value = |i: usize, o: usize| ((i * 23 + o * 7 + i / 11) % 211) as f32 * 0.00373 - 0.39;
    for dtype in [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::Q8_0,
        DType::Q4_0,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
    ] {
        for ggml in [false, true] {
            for output_rows in [false, true] {
                // row_major is always row-major in the QIF logical axes. GGML
                // compatibility reverses those axes in the packed kernel view.
                let physical_rows = ggml || output_rows;
                let physical: Vec<_> = if physical_rows {
                    (0..columns)
                        .flat_map(|o| (0..inner).map(move |i| value(i, o)))
                        .collect()
                } else {
                    (0..inner)
                        .flat_map(|i| (0..columns).map(move |o| value(i, o)))
                        .collect()
                };
                let (reader, source) =
                    fixture(dtype, &[inner, columns], &physical, ggml, output_rows);
                let raw = crate::ggml::load_ggml_as_float(&source, "weight").unwrap();
                let actual = load_linear(&reader, "weight").unwrap();
                assert_eq!(actual.shape().dims(), &[inner, columns]);
                assert_eq!(
                    reader.tensor_bytes("weight").unwrap(),
                    source.tensor_bytes("weight").unwrap()
                );
                match &actual {
                    LinearWeight::Q8_0(w) | LinearWeight::Q8_0OutputRows(w) => {
                        assert_eq!(w, &source.tensor_q8_0("weight").unwrap())
                    }
                    LinearWeight::Q4_0(w) | LinearWeight::Q4_0OutputRows(w) => {
                        assert_eq!(w, &source.tensor_q4_0("weight").unwrap())
                    }
                    LinearWeight::Q4_K(w) | LinearWeight::Q4_KOutputRows(w) => {
                        assert_eq!(w, &source.tensor_q4_k("weight").unwrap())
                    }
                    LinearWeight::Q5_K(w) | LinearWeight::Q5_KOutputRows(w) => {
                        assert_eq!(w, &source.tensor_q5_k("weight").unwrap())
                    }
                    LinearWeight::Q6_K(w) | LinearWeight::Q6_KOutputRows(w) => {
                        assert_eq!(w, &source.tensor_q6_k("weight").unwrap())
                    }
                    _ => {}
                }
                let output = actual.matmul(&CpuScalar::new(), &input_tensor).unwrap();
                for r in 0..rows {
                    for o in 0..columns {
                        let expected: f64 = (0..inner)
                            .map(|i| {
                                let index = if physical_rows {
                                    o * inner + i
                                } else {
                                    i * columns + o
                                };
                                f64::from(input[r * inner + i])
                                    * f64::from(raw.as_f32().unwrap()[index])
                            })
                            .sum();
                        let actual = f64::from(output.as_f32().unwrap()[r * columns + o]);
                        assert!((actual-expected).abs() < 1e-4 + expected.abs()*1e-5,
                            "{dtype:?} ggml={ggml} layout_rows={output_rows} at [{r},{o}]: {actual} != {expected}");
                    }
                }
            }
        }
    }
}

#[test]
fn native_output_rows_reject_unaligned_packed_inner_width() {
    let (reader, _) = fixture(DType::Q8_0, &[31, 32], &[0.25; 31 * 32], false, true);
    let error = load_linear(&reader, "weight").unwrap_err().to_string();
    assert!(error.contains("input width 31"), "{error}");
}
