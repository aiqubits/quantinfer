//! Exhaustive scalar oracle for the codebook that is actually serialized.
use super::{quantize_q4_k, quantize_q5_k, quantize_q6_k};
use crate::{NativeQ4KProvider, NativeQ5KProvider, NativeQ6KProvider, QuantizationProvider};
use half::f16;
use quantinfer_core::Tensor;

fn input() -> Tensor {
    let mut values = Vec::new();
    for block in 0..4 {
        for index in 0..256 {
            let value = ((index * 17 + block * 31) % 67) as f32 - 33.0;
            values.push(
                value / 19.0 * (block + 1) as f32 * 0.031 * (0.27 + (index / 32) as f32 * 0.09)
                    - 0.012,
            );
        }
    }
    values.extend([0.0; 256]);
    values.extend([-0.002; 256]);
    values.extend([1.0e-8; 256]);
    Tensor::from_f32(vec![values.len()], values).unwrap()
}
fn scale_min(bytes: &[u8; 12], sub: usize) -> (f64, f64) {
    let (s, m) = if sub < 4 {
        (bytes[sub] & 63, bytes[4 + sub] & 63)
    } else {
        let index = sub - 4;
        (
            (bytes[8 + index] & 15) | ((bytes[index] >> 6) << 4),
            (bytes[8 + index] >> 4) | ((bytes[index + 4] >> 6) << 4),
        )
    };
    (f64::from(s), f64::from(m))
}
fn best(value: f32, actual: i32, range: std::ops::RangeInclusive<i32>, scale: f64, min: f64) {
    let error = |code: i32| (scale * f64::from(code) - min - f64::from(value)).abs();
    let expected = range.map(error).fold(f64::INFINITY, f64::min);
    let tolerance = (scale.abs() * 64.0 + min.abs() + f64::from(value).abs())
        * f64::from(f32::EPSILON)
        + 1.0e-300;
    assert!(
        error(actual) <= expected + tolerance,
        "value {value} code {actual} error {} best {expected} scale {scale} min {min}",
        error(actual)
    );
}

#[test]
fn q4_k_uses_nearest_points_of_its_stored_f16_codebook() {
    let input = input();
    let packed = quantize_q4_k(&input).unwrap();
    for (values, block) in input
        .as_f32()
        .unwrap()
        .as_chunks::<256>()
        .0
        .iter()
        .zip(packed.blocks())
    {
        let wire_scale = f16::from_f32(block.scale).to_f32();
        let wire_min = f16::from_f32(block.min_scale).to_f32();
        for (index, &value) in values.iter().enumerate() {
            let (s, m) = scale_min(&block.scales, index / 32);
            let byte = block.values[index / 64 * 32 + index % 32];
            let code = if index % 64 < 32 {
                byte & 15
            } else {
                byte >> 4
            };
            best(
                value,
                i32::from(code),
                0..=15,
                f64::from(wire_scale) * s,
                f64::from(wire_min) * m,
            );
        }
        assert_eq!(block.scale, wire_scale);
        assert_eq!(block.min_scale, wire_min);
    }
}
#[test]
fn q5_k_uses_nearest_points_of_its_stored_f16_codebook() {
    let input = input();
    let packed = quantize_q5_k(&input).unwrap();
    for (values, block) in input
        .as_f32()
        .unwrap()
        .as_chunks::<256>()
        .0
        .iter()
        .zip(packed.blocks())
    {
        let wire_scale = f16::from_f32(block.scale).to_f32();
        let wire_min = f16::from_f32(block.min_scale).to_f32();
        for (index, &value) in values.iter().enumerate() {
            let (s, m) = scale_min(&block.scales, index / 32);
            let group = index / 64;
            let within = index % 64;
            let byte = block.values[group * 32 + within % 32];
            let low = if within < 32 { byte & 15 } else { byte >> 4 };
            let high = (block.high_bits[within % 32] >> (group * 2 + within / 32)) & 1;
            best(
                value,
                i32::from(low | (high << 4)),
                0..=31,
                f64::from(wire_scale) * s,
                f64::from(wire_min) * m,
            );
        }
        assert_eq!(block.scale, wire_scale);
        assert_eq!(block.min_scale, wire_min);
    }
}
#[test]
fn q6_k_uses_nearest_points_of_its_stored_f16_codebook() {
    let input = input();
    let packed = quantize_q6_k(&input).unwrap();
    for (values, block) in input
        .as_f32()
        .unwrap()
        .as_chunks::<256>()
        .0
        .iter()
        .zip(packed.blocks())
    {
        let wire_scale = f16::from_f32(block.scale).to_f32();
        for (index, &value) in values.iter().enumerate() {
            let half = index / 128;
            let stripe = index % 128 / 32;
            let column = index % 32;
            let low = block.low_bits[half * 64 + stripe % 2 * 32 + column];
            let low = if stripe < 2 { low & 15 } else { low >> 4 };
            let high = (block.high_bits[half * 32 + column] >> (stripe * 2)) & 3;
            best(
                value,
                i32::from(low | (high << 4)) - 32,
                -32..=31,
                f64::from(wire_scale) * f64::from(block.scales[index / 16]),
                0.0,
            );
        }
        assert_eq!(block.scale, wire_scale);
    }
}

#[test]
fn k_quant_providers_reject_nonrepresentable_wire_scales() {
    let input = Tensor::from_f32(vec![256], vec![f32::MAX; 256]).unwrap();
    let providers: [&dyn QuantizationProvider; 3] =
        [&NativeQ4KProvider, &NativeQ5KProvider, &NativeQ6KProvider];
    for provider in providers {
        assert!(provider.quantize(&input).is_err());
    }
}

#[test]
fn k_quant_reports_subgrid_loss() {
    let input = Tensor::from_f32(vec![256], vec![1.0e-8; 256]).unwrap();
    let providers: [&dyn QuantizationProvider; 3] =
        [&NativeQ4KProvider, &NativeQ5KProvider, &NativeQ6KProvider];
    for provider in providers {
        let packed = provider.quantize(&input).unwrap();
        assert!(provider
            .dequantize(&packed)
            .unwrap()
            .as_f32()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0));
        assert_eq!(
            provider
                .error_report(&input, &packed)
                .unwrap()
                .relative_l2_error,
            1.0
        );
    }
}

#[test]
fn k_quant_keeps_values_resolvable_by_subnormal_f16_scales() {
    let input = Tensor::from_f32(vec![256], vec![5.0e-7; 256]).unwrap();
    let providers: [&dyn QuantizationProvider; 3] =
        [&NativeQ4KProvider, &NativeQ5KProvider, &NativeQ6KProvider];
    for provider in providers {
        let packed = provider.quantize(&input).unwrap();
        assert!(
            provider
                .error_report(&input, &packed)
                .unwrap()
                .relative_l2_error
                < 0.1
        );
    }
}
