use super::{dequantize, dequantize_with_zero_point_layout, GptqConfig, ZeroPointLayout};
use quantinfer_format::SafeTensorFile;
use serde_json::json;

fn fixture(entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> SafeTensorFile {
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    for (name, dtype, shape, payload) in entries {
        let end = offset + payload.len();
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let header = serde_json::to_vec(&json!(header)).unwrap();
    let mut bytes = Vec::with_capacity(8 + header.len() + offset);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    for (_, _, _, payload) in entries {
        bytes.extend_from_slice(payload);
    }
    SafeTensorFile::open(bytes).unwrap()
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn decodes_standard_packed_gptq_matrix() {
    let bits = 4;
    let mut qweight = [0_u32; 2];
    for (input, code) in [2_u32, 3, 4, 5, 6, 7, 8, 9].into_iter().enumerate() {
        qweight[0] |= code << (input * bits);
    }
    for (input, code) in [1_u32, 2, 3, 4, 5, 6, 7, 8].into_iter().enumerate() {
        qweight[1] |= code << (input * bits);
    }
    let file = fixture(&[
        ("layer.qweight", "I32", vec![1, 2], words(&qweight)),
        ("layer.qzeros", "I32", vec![1, 2], words(&[33, 33])),
        (
            "layer.scales",
            "F32",
            vec![2, 2],
            floats(&[1.0, 2.0, 3.0, 4.0]),
        ),
    ]);
    let tensor = dequantize(&file, "layer", &GptqConfig::new(bits, 4, false)).unwrap();
    assert_eq!(tensor.shape().dims(), &[8, 2]);
    assert_eq!(
        tensor.as_f32().unwrap(),
        &[
            0.0, -4.0, 1.0, -2.0, 2.0, 0.0, 3.0, 2.0, 12.0, 8.0, 15.0, 12.0, 18.0, 16.0, 21.0,
            20.0,
        ]
    );
}

#[test]
fn uses_g_idx_and_symmetric_midpoint_when_qzeros_are_absent() {
    let mut qword = 0_u32;
    for input in 0..8 {
        qword |= (input as u32) << (input * 4);
    }
    let g_idx: Vec<u32> = vec![1; 8];
    let file = fixture(&[
        ("layer.qweight", "U32", vec![1, 1], words(&[qword])),
        ("layer.scales", "F32", vec![2, 1], floats(&[1.0, 2.0])),
        ("layer.g_idx", "I32", vec![8], words(&g_idx)),
    ]);
    let tensor = dequantize(&file, "layer", &GptqConfig::new(4, 4, true)).unwrap();
    assert_eq!(
        tensor.as_f32().unwrap(),
        &[-16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0]
    );
}

#[test]
fn rejects_invalid_bits_and_shapes() {
    assert!(GptqConfig::new(3, 4, false).validate().is_err());
}

#[test]
fn all_bit_widths_and_zero_layouts_match_an_independent_modular_reference() {
    let input = 256;
    let group_size = 32;
    let groups = input / group_size;
    for bits in [2, 4, 8] {
        let pack = 32 / bits;
        let output = groups * pack; // A deliberately ambiguous square packed table.
        let modulus = 1_i64 << bits;
        let code = |i: usize, o: usize| ((i * 7 + o * 11 + 2) % modulus as usize) as u32;
        let zero = |g: usize, o: usize| ((g * 3 + o * 5) % modulus as usize) as u32;
        let scale = |g: usize, o: usize| ((g + o) % 3 + 1) as f32 * 0.25;
        let mut qweight = vec![0_u32; input / pack * output];
        for i in 0..input {
            for o in 0..output {
                qweight[i / pack * output + o] |= code(i, o) << (i % pack * bits);
            }
        }
        let expected: Vec<_> = (0..input)
            .flat_map(|i| {
                (0..output).map(move |o| {
                    let g = i / group_size;
                    ((f64::from(code(i, o)) - f64::from(zero(g, o))) * f64::from(scale(g, o)))
                        as f32
                })
            })
            .collect();
        let scales: Vec<_> = (0..groups)
            .flat_map(|g| (0..output).map(move |o| scale(g, o)))
            .collect();
        for offset in [0, 1, -1, 257] {
            for layout in [
                ZeroPointLayout::GroupPacked,
                ZeroPointLayout::PackedGroup,
                ZeroPointLayout::GroupDirect,
                ZeroPointLayout::DirectGroup,
            ] {
                let packed = matches!(
                    layout,
                    ZeroPointLayout::GroupPacked | ZeroPointLayout::PackedGroup
                );
                let transposed = matches!(
                    layout,
                    ZeroPointLayout::PackedGroup | ZeroPointLayout::DirectGroup
                );
                let columns = if packed { output / pack } else { output };
                let shape = if transposed {
                    vec![columns, groups]
                } else {
                    vec![groups, columns]
                };
                let mut zeros = vec![0_u32; groups * columns];
                for g in 0..groups {
                    for o in 0..output {
                        let stored =
                            (i64::from(zero(g, o)) - i64::from(offset)).rem_euclid(modulus) as u32;
                        let c = if packed { o / pack } else { o };
                        let index = if transposed {
                            c * groups + g
                        } else {
                            g * columns + c
                        };
                        zeros[index] |= stored << if packed { o % pack * bits } else { 0 };
                    }
                }
                let file = fixture(&[
                    (
                        "layer.qweight",
                        "I32",
                        vec![input / pack, output],
                        words(&qweight),
                    ),
                    ("layer.qzeros", "I32", shape, words(&zeros)),
                    ("layer.scales", "F32", vec![groups, output], floats(&scales)),
                ]);
                let mut config = GptqConfig::new(bits, group_size, false);
                config.zero_offset = offset;
                let actual =
                    dequantize_with_zero_point_layout(&file, "layer", &config, layout).unwrap();
                assert_eq!(
                    actual.as_f32().unwrap(),
                    expected,
                    "bits={bits} offset={offset} layout={layout:?}"
                );
                if layout == ZeroPointLayout::GroupPacked {
                    assert_eq!(dequantize(&file, "layer", &config).unwrap(), actual);
                    assert!(dequantize_with_zero_point_layout(
                        &file,
                        "layer",
                        &config,
                        ZeroPointLayout::GroupDirect
                    )
                    .unwrap_err()
                    .contains("explicit"));
                }
            }
        }
    }
}
#[test]
fn standard_square_zero_point_groups_are_not_transposed() {
    // Standard GPTQ: qzeros[group, packed_output]. All matrix widths are multiples of 32.
    let zeros: Vec<u32> = (0..4)
        .flat_map(|g| {
            (0..4).map(move |c| {
                let code = ((g * 3 + c) % 7 + 1) as u32;
                code * 0x1111_1111
            })
        })
        .collect();
    let file = fixture(&[
        (
            "layer.qweight",
            "I32",
            vec![16, 32],
            words(&[0x9999_9999; 512]),
        ),
        ("layer.qzeros", "I32", vec![4, 4], words(&zeros)),
        ("layer.scales", "F32", vec![4, 32], floats(&[1.0; 128])),
    ]);
    let actual = dequantize(&file, "layer", &GptqConfig::new(4, 32, false)).unwrap();
    let expected: Vec<f32> = (0..128)
        .flat_map(|input| {
            (0..32).map(move |output| {
                let packed_zero = ((input / 32 * 3 + output / 8) % 7 + 1) as u32;
                9.0 - (packed_zero + 1) as f32
            })
        })
        .collect();
    let first = actual
        .as_f32()
        .unwrap()
        .iter()
        .zip(&expected)
        .position(|(a, b)| a != b);
    if let Some(index) = first {
        eprintln!(
            "GPTQ zero orientation: input={} output={} actual={} expected={}",
            index / 32,
            index % 32,
            actual.as_f32().unwrap()[index],
            expected[index]
        );
    }
    assert!(
        first.is_none(),
        "standard square qzeros was interpreted as its transpose"
    );
}

#[test]
fn standard_zero_offset_wraps_within_its_bit_width() {
    let file = fixture(&[
        (
            "layer.qweight",
            "I32",
            vec![4, 32],
            words(&[0x2222_2222; 128]),
        ),
        ("layer.qzeros", "I32", vec![1, 4], words(&[0xffff_ffff; 4])),
        ("layer.scales", "F32", vec![1, 32], floats(&[1.0; 32])),
    ]);
    let actual = dequantize(&file, "layer", &GptqConfig::new(4, 32, false)).unwrap();
    eprintln!(
        "GPTQ packed-zero wrap: actual={} expected=2",
        actual.as_f32().unwrap()[0]
    );
    assert!(
        actual.as_f32().unwrap().iter().all(|v| *v == 2.0),
        "stored code 15 + 1 must wrap to logical zero point 0 for standard 4-bit GPTQ"
    );
}

#[test]
fn finite_scale_must_not_produce_infinite_weights() {
    let file = fixture(&[
        (
            "layer.qweight",
            "I32",
            vec![4, 32],
            words(&[0x3333_3333; 128]),
        ),
        ("layer.qzeros", "I32", vec![1, 4], words(&[0; 4])),
        ("layer.scales", "F32", vec![1, 32], floats(&[f32::MAX; 32])),
    ]);
    let result = dequantize(&file, "layer", &GptqConfig::new(4, 32, false));
    if let Ok(tensor) = &result {
        eprintln!("GPTQ decoded finite scale: {}", tensor.as_f32().unwrap()[0]);
    }
    let error = result.unwrap_err();
    assert!(
        error.contains("layer") && error.contains("[0, 0]") && error.contains("finite F32"),
        "{error}"
    );
}
