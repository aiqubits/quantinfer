use super::{dequantize_file, AutoRoundConfig};
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
fn parses_autoround_aliases_and_dequantizes_gptq_layout() {
    let config = AutoRoundConfig::from_json(&json!({
        "quant_method": "auto_round",
        "w_bit": 4,
        "q_group_size": 8,
        "zero_point": true
    }))
    .unwrap();
    assert_eq!(config, AutoRoundConfig::new(4, 8, false));
    let file = fixture(&[
        ("layer.qweight", "I32", vec![1, 1], words(&[0x7654_3210])),
        ("layer.qzeros", "I32", vec![1, 1], words(&[0x0000_0000])),
        ("layer.scales", "F32", vec![1, 1], floats(&[2.0])),
    ]);
    let tensor = dequantize_file(&file, "layer", &config).unwrap();
    assert_eq!(
        tensor.as_f32().unwrap(),
        &[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
}

#[test]
fn rejects_unknown_method_and_invalid_values() {
    assert!(AutoRoundConfig::from_json(&json!({
        "quant_method": "awq", "bits": 4, "group_size": 128
    }))
    .is_err());
    assert!(AutoRoundConfig::from_json(&json!({
        "bits": 3, "group_size": 128
    }))
    .is_err());
}

#[test]
fn aliases_validate_present_values_and_preserve_valid_defaults() {
    let defaults = json!({"bits":4,"group_size":32});
    assert_eq!(
        AutoRoundConfig::from_json(&defaults).unwrap(),
        AutoRoundConfig::new(4, 32, false)
    );
    for (key, value) in [
        ("quantization_method", json!(false)),
        ("w_bit", json!(8)),
        ("q_group_size", json!("32")),
        ("zero_offset", json!(null)),
        ("zero_offset", json!(i64::MIN)),
        ("zero_offset", json!(2147483648_i64)),
        ("sym", json!("false")),
        ("zero_point", json!(0)),
    ] {
        let mut value_with_alias = defaults.clone();
        value_with_alias[key] = value;
        assert!(
            AutoRoundConfig::from_json(&value_with_alias).is_err(),
            "{value_with_alias}"
        );
    }
    let valid = json!({"bits":4,"w_bit":4,"group_size":32,"q_group_size":32,
        "sym":true,"symmetric":true,"zero_point":false,"zero_offset":0,
        "quant_method":"auto_round","quantization_method":"GPTQ"});
    let parsed = AutoRoundConfig::from_json(&valid).unwrap();
    assert!(parsed.sym);
    assert_eq!(parsed.zero_offset, 0);
}

#[test]
fn explicit_layout_and_standard_zero_wrap_reach_the_gptq_decoder() {
    let file = fixture(&[
        (
            "layer.qweight",
            "I32",
            vec![1, 16],
            words(&[0x9999_9999; 16]),
        ),
        (
            "layer.qzeros",
            "I32",
            vec![2, 2],
            words(&[0xffff_ffff, 0x1111_1111, 0x2222_2222, 0x3333_3333]),
        ),
        ("layer.scales", "F32", vec![2, 16], floats(&[1.0; 32])),
    ]);
    let config = AutoRoundConfig::new(4, 4, false);
    let standard = dequantize_file(&file, "layer", &config).unwrap();
    let transposed = super::dequantize_with_zero_point_layout(
        &file,
        "layer",
        &config,
        super::ZeroPointLayout::PackedGroup,
    )
    .unwrap();
    assert_eq!(standard.as_f32().unwrap()[0], 9.0); // (15 + 1) mod 16 = 0.
    assert_eq!(standard.as_f32().unwrap()[8], 7.0);
    assert_eq!(transposed.as_f32().unwrap()[8], 6.0);
    assert_eq!(
        super::dequantize_with_zero_point_layout(
            &file,
            "layer",
            &config,
            super::ZeroPointLayout::GroupPacked
        )
        .unwrap(),
        standard
    );
}
#[test]
fn present_malformed_decode_settings_are_not_missing_defaults() {
    let mut accepted = Vec::new();
    for (name, value) in [
        ("quant_method", json!(7)),
        ("zero_offset", json!("0")),
        ("zero_offset", json!(1.5)),
        ("zero_offset", json!(u64::MAX)),
    ] {
        let mut input = json!({"bits":4,"group_size":8});
        input[name] = value;
        let result = AutoRoundConfig::from_json(&input);
        if let Ok(config) = result {
            accepted.push(format!("{input} -> {config:?}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "present invalid fields silently defaulted: {accepted:?}"
    );
}
