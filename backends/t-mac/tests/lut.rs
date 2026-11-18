use quantinfer_backend_tmac::{Lut4Weights, NativeLut4Kernel, TmacBackend};
use quantinfer_core::{Q4_0Block, QuantizedTensor4, Tensor};
use quantinfer_kernel::KernelBackend;

#[test]
fn scalar_and_simd_lut_matrices_match_an_independent_f64_dot_product() {
    let native = NativeLut4Kernel::new();
    let scalar = NativeLut4Kernel::scalar();
    for (m, n, k) in [(1, 1, 32), (3, 5, 64), (17, 9, 96), (2, 3, 128)] {
        let rhs = QuantizedTensor4::from_blocks(
            vec![k, n],
            (0..k * n / 32)
                .map(|block| Q4_0Block {
                    scale: [0.0, 0.1, -0.37, 0.5][block % 4],
                    values: std::array::from_fn(|index| {
                        ((index * 7 + index / 4 + block * 3) % 16) as u8
                    }),
                })
                .collect(),
        )
        .unwrap();
        let packed = Lut4Weights::from_output_rows(&rhs).unwrap();
        assert_eq!(packed.shape(), [k, n]);
        assert_eq!(packed.packed_bytes(), k * n / 2 + k * n / 32 * 4);
        let values: Vec<_> = (0..m * k)
            .map(|index| (index % 23) as f32 * 0.173 - 1.3)
            .collect();
        let lhs = Tensor::from_f32(vec![m, k], values.clone()).unwrap();
        for kernel in [&native, &scalar] {
            let actual = kernel.matmul_prepared(&lhs, &packed).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0f64;
                    let mut magnitude = 0.0f64;
                    for channel in 0..k {
                        let block = &rhs.blocks()[col * (k / 32) + channel / 32];
                        let product = f64::from(values[row * k + channel])
                            * (f64::from(block.values[channel % 32]) - 8.0)
                            * f64::from(block.scale);
                        expected += product;
                        magnitude += product.abs();
                    }
                    let difference =
                        (f64::from(actual.as_f32().unwrap()[row * n + col]) - expected).abs();
                    assert!(
                        difference <= 512.0 * f64::from(f32::EPSILON) * magnitude + 1e-6,
                        "{m}x{k}x{n} [{row},{col}] error {difference}"
                    );
                }
            }
        }
        let backend = TmacBackend::open().unwrap();
        assert_eq!(
            backend
                .matmul_q4_0_output_rows(&lhs, &rhs)
                .unwrap()
                .as_f32()
                .unwrap(),
            native
                .matmul_prepared(&lhs, &packed)
                .unwrap()
                .as_f32()
                .unwrap()
        );
    }
}

#[test]
fn prepared_weights_are_immutable_snapshots_and_invalid_profiles_fail() {
    let mut rhs = QuantizedTensor4::from_blocks(
        vec![32, 1],
        vec![Q4_0Block {
            scale: 0.1,
            values: [9; 32],
        }],
    )
    .unwrap();
    let prepared = Lut4Weights::from_output_rows(&rhs).unwrap();
    rhs.blocks_mut()[0].values.fill(10);
    let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
    let native = NativeLut4Kernel::new();
    assert!(
        (native
            .matmul_prepared(&lhs, &prepared)
            .unwrap()
            .as_f32()
            .unwrap()[0]
            - 3.2)
            .abs()
            < 1e-6
    );
    assert!(
        (TmacBackend::open()
            .unwrap()
            .matmul_q4_0_output_rows(&lhs, &rhs)
            .unwrap()
            .as_f32()
            .unwrap()[0]
            - 6.4)
            .abs()
            < 1e-6
    );
    rhs.blocks_mut()[0].values[0] = 16;
    assert!(Lut4Weights::from_output_rows(&rhs).is_err());
    rhs.blocks_mut()[0].values[0] = 0;
    rhs.blocks_mut()[0].scale = f32::NAN;
    assert!(Lut4Weights::from_output_rows(&rhs).is_err());
    let bad = QuantizedTensor4::from_blocks(
        vec![16, 2],
        vec![Q4_0Block {
            scale: 1.0,
            values: [8; 32],
        }],
    )
    .unwrap();
    assert!(Lut4Weights::from_output_rows(&bad).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::MAX] {
        let input = Tensor::from_f32(vec![1, 32], vec![value; 32]).unwrap();
        assert!(native.matmul_prepared(&input, &prepared).is_err());
    }
}
