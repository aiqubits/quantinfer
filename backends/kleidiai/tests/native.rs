#![cfg(all(
    feature = "native",
    target_arch = "aarch64",
    target_os = "linux",
    target_endian = "little"
))]
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_backend_kleidiai::{KleidiAiBackend, NativeKleidiAiKernel};
use quantinfer_core::{Q4_0Block, QuantizedTensor4, Tensor};
use quantinfer_kernel::KernelBackend;

#[test]
fn real_sdk_dotprod_matches_scalar_layout_and_quantization_bounds() {
    let backend = KleidiAiBackend::open().expect("native conformance requires AArch64 dotprod");
    let cpu = CpuScalar::new();
    for (m, n, k) in [(1, 1, 32), (3, 5, 64), (4, 4, 32), (17, 9, 96), (2, 3, 64)] {
        let blocks: Vec<_> = (0..n * k / 32)
            .map(|block| Q4_0Block {
                scale: [0.125, -0.25, 0.0, 0.5][block % 4],
                values: std::array::from_fn(|index| {
                    ((index * 7 + index / 4 + block * 3) % 16) as u8
                }),
            })
            .collect();
        let rhs = QuantizedTensor4::from_blocks(vec![k, n], blocks).unwrap();
        for fractional in [false, true] {
            let values: Vec<_> = (0..m * k)
                .map(|index| {
                    if fractional {
                        (index % 17) as f32 * 0.173 - 1.1
                    } else if index % 32 == 0 {
                        127.0 / 64.0
                    } else {
                        ((index * 13 + index / 32 * 7) % 255) as f32 / 64.0 - 127.0 / 64.0
                    }
                })
                .collect();
            let lhs = Tensor::from_f32(vec![m, k], values.clone()).unwrap();
            let actual = backend.matmul_q4_0_output_rows(&lhs, &rhs).unwrap();
            let expected = cpu.matmul_q4_0_output_rows(&lhs, &rhs).unwrap();
            assert_eq!(actual.shape().dims(), [m, n]);
            for row in 0..m {
                for col in 0..n {
                    let mut bound = 0.0f32;
                    let mut magnitude = 0.0f32;
                    for group in 0..k / 32 {
                        let x = &values[row * k + group * 32..row * k + (group + 1) * 32];
                        let weight = &rhs.blocks()[col * (k / 32) + group];
                        let maximum = x.iter().copied().map(f32::abs).fold(0.0, f32::max);
                        let scale = maximum / 127.0;
                        let rounded = half::f16::from_f32(scale).to_f32();
                        let quantization_bound = if fractional {
                            rounded * 0.5 + (rounded - scale).abs() * 127.0
                        } else {
                            0.0
                        };
                        for (&x, &code) in x.iter().zip(&weight.values) {
                            let weight = (code as f32 - 8.0) * weight.scale;
                            bound += quantization_bound * weight.abs();
                            magnitude += (x * weight).abs();
                        }
                    }
                    let difference = (actual.as_f32().unwrap()[row * n + col]
                        - expected.as_f32().unwrap()[row * n + col])
                        .abs();
                    let tolerance = bound + 64.0 * f32::EPSILON * magnitude.max(1.0);
                    assert!(
                        difference <= tolerance,
                        "{m}x{k}x{n}, [{row},{col}], difference {difference}, limit {tolerance}"
                    );
                }
            }
        }
    }
    println!(
        "verified KleidiAI dotprod {}",
        NativeKleidiAiKernel::sdk_revision()
    );
}

#[test]
fn native_rejects_unrepresentable_inputs_before_sdk_conversion() {
    let backend = KleidiAiBackend::open().unwrap();
    let rhs = QuantizedTensor4::from_blocks(
        vec![32, 1],
        vec![Q4_0Block {
            scale: 0.25,
            values: [9; 32],
        }],
    )
    .unwrap();
    for value in [f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
        let lhs = Tensor::from_f32(vec![1, 32], vec![value; 32]).unwrap();
        assert!(backend.matmul_q4_0_output_rows(&lhs, &rhs).is_err());
    }
    let mut incompatible = rhs.clone();
    incompatible.blocks_mut()[0].scale = 0.1;
    let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
    assert!(backend
        .matmul_q4_0_output_rows(&lhs, &incompatible)
        .is_err());
}
