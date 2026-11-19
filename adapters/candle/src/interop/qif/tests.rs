use super::*;
use crate::from_candle;
use quantinfer_core::{QuantizationScheme, Shape, TensorStorageDescriptor};
use quantinfer_format::{qif, ModelManifest, TensorManifest};
use std::collections::BTreeMap;

fn publish(
    raw: &[u8],
    dtype: DType,
    rows: usize,
    columns: usize,
    output_rows: bool,
    ggml: bool,
    codec: qif::Codec,
) -> Vec<u8> {
    let (scheme, size) = match dtype {
        DType::Q8_0 => (QuantizationScheme::Q8_0, 32),
        DType::Q4_0 => (QuantizationScheme::Q4_0, 32),
        DType::Q4_K => (QuantizationScheme::Q4_K, 256),
        DType::Q5_K => (QuantizationScheme::Q5_K, 256),
        DType::Q6_K => (QuantizationScheme::Q6_K, 256),
        _ => unreachable!(),
    };
    let mut descriptor =
        TensorStorageDescriptor::quantized(dtype, scheme, Shape::from(vec![rows, columns]), size)
            .unwrap();
    if output_rows {
        descriptor.physical_layout = quantinfer_format::GGML_OUTPUT_ROWS_LAYOUT.into();
    }
    let mut manifest = ModelManifest::new("candle-logical-layout", "fixture");
    manifest.quantization = scheme;
    for name in ["weight", "alias"] {
        manifest.tensors.push(TensorManifest {
            name: name.into(),
            descriptor: descriptor.clone(),
            checksum_sha256: None,
        });
    }
    let metadata = BTreeMap::from([(
        "quantinfer.qif.ggml_dimension_order".into(),
        qif::MetadataValue::Bool(ggml),
    )]);
    let mut bytes = Vec::new();
    qif::write_qif_with_codec(manifest, &metadata, |_| Ok(raw), &mut bytes, codec).unwrap();
    bytes
}

#[test]
fn original_output_row_counterexample_preserves_every_logical_value() {
    for columns in [3, 64] {
        let value = |input: usize, output: usize| ((input * 3 + output * 5) % 31) as i8 - 15;
        let mut raw = Vec::new();
        for output in 0..columns {
            raw.extend_from_slice(&0x3c00u16.to_le_bytes());
            raw.extend((0..32).map(|input| value(input, output) as u8));
        }
        let bytes = publish(
            &raw,
            DType::Q8_0,
            32,
            columns,
            true,
            false,
            qif::Codec::None,
        );
        let reader = qif::Reader::open(&bytes).unwrap();
        reader.verify().unwrap();
        assert!(packed_from_qif(&reader, "weight", &Device::Cpu).is_err());
        let view = packed_view_from_qif(&reader, "weight", &Device::Cpu).unwrap();
        assert_eq!(view.storage().data().unwrap().as_ref(), raw);
        assert_eq!(view.logical_shape(), [32, columns]);
        assert!(view.is_transposed());
        let logical = from_candle(&view.dequantize(&Device::Cpu).unwrap()).unwrap();
        let values = logical.as_f32().unwrap();
        assert_eq!(values[1], -10.);
        for input in 0..32 {
            for output in 0..columns {
                assert_eq!(
                    values[input * columns + output],
                    f32::from(value(input, output))
                );
            }
        }
    }
}

fn block(dtype: DType, index: usize) -> (Vec<u8>, f32, usize) {
    let c = (index % 15 + 1) as u8;
    match dtype {
        DType::Q8_0 => {
            let v = ((index * 7) % 127) as i8 - 63;
            let mut raw = 0x3c00u16.to_le_bytes().to_vec();
            raw.extend([v as u8; 32]);
            (raw, f32::from(v), 32)
        }
        DType::Q4_0 => {
            let mut raw = 0x3c00u16.to_le_bytes().to_vec();
            raw.extend([c | c << 4; 16]);
            (raw, f32::from(c) - 8., 32)
        }
        DType::Q4_K | DType::Q5_K => {
            let mut raw = vec![0, 0x3c, 0, 0];
            // All eight local scales are one, minima zero.
            raw.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
            let high = usize::from(dtype == DType::Q5_K && index % 2 == 1);
            if dtype == DType::Q5_K {
                raw.extend([if high == 1 { 255 } else { 0 }; 32]);
            }
            raw.extend([c | c << 4; 128]);
            (raw, f32::from(c) + 16. * high as f32, 256)
        }
        DType::Q6_K => {
            let mut raw = vec![c | c << 4; 128];
            raw.extend([0x55; 64]);
            raw.extend([1; 16]);
            raw.extend([0, 0x3c]);
            (raw, f32::from(c) - 16., 256)
        }
        _ => unreachable!(),
    }
}

#[test]
fn all_packed_encodings_layouts_codecs_and_dimension_profiles_preserve_coordinates() {
    let (rows, columns) = (256, 512);
    for dtype in [
        DType::Q8_0,
        DType::Q4_0,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
    ] {
        let size = block(dtype, 0).2;
        let mut raw = Vec::new();
        let mut values = Vec::new();
        for index in 0..rows * columns / size {
            let (bytes, value, _) = block(dtype, index);
            raw.extend(bytes);
            values.push(value);
        }
        for ggml in [false, true] {
            for output_rows in [false, true] {
                for codec in [qif::Codec::None, qif::Codec::Zstd, qif::Codec::Lz4] {
                    let bytes = publish(&raw, dtype, rows, columns, output_rows, ggml, codec);
                    let reader = qif::Reader::open(&bytes).unwrap();
                    reader.verify().unwrap();
                    assert_eq!(reader.tensor_info("alias").unwrap().alias_of, Some(0));
                    let transposed = output_rows && !ggml;
                    assert_eq!(
                        packed_from_qif(&reader, "weight", &Device::Cpu).is_ok(),
                        !transposed
                    );
                    for name in ["weight", "alias"] {
                        let view = packed_view_from_qif(&reader, name, &Device::Cpu).unwrap();
                        assert_eq!(view.logical_shape(), [rows, columns]);
                        assert_eq!(view.is_transposed(), transposed);
                        assert_eq!(view.storage().data().unwrap().as_ref(), raw);
                        let decoded = from_candle(&view.dequantize(&Device::Cpu).unwrap()).unwrap();
                        let actual = decoded.as_f32().unwrap();
                        for r in 0..rows {
                            for c in 0..columns {
                                let physical = if transposed {
                                    c * rows + r
                                } else {
                                    r * columns + c
                                };
                                assert_eq!(
                                    actual[r * columns + c],
                                    values[physical / size],
                                    "{dtype:?}/{ggml}/{output_rows}/{codec:?} at {r},{c}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rejects_finite_payload_corruption_before_exposing_candle_storage() {
    let (raw, _, _) = block(DType::Q8_0, 0);
    let mut bytes = publish(&raw, DType::Q8_0, 1, 32, false, false, qif::Codec::None);
    let payload = u64::from_le_bytes(bytes[72..80].try_into().unwrap()) as usize;
    bytes[payload + raw.len() - 1] ^= 1;
    let reader = qif::Reader::open(&bytes).unwrap();
    assert!(reader.verify().is_err());
    for name in ["weight", "alias"] {
        assert!(
            matches!(packed_view_from_qif(&reader,name,&Device::Cpu),Err(InteropError::Invalid(message)) if message.contains("checksum"))
        );
    }
}
