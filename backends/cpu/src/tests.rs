use super::{CpuIsa, CpuRuntimeConfig, CpuScalar};
use quantinfer_core::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, QuantizedTensorQ5K, Tensor,
};
use quantinfer_kernel::KernelBackend;

#[test]
fn computes_matrix_product() {
    let backend = CpuScalar::new();
    let lhs = Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Tensor::from_f32(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = backend.matmul(&lhs, &rhs).unwrap();
    assert_eq!(result.as_f32().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn computes_q8_matrix_product() {
    let backend = CpuScalar::new();
    let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
    let mut values = [0_i8; 32];
    values.fill(1);
    let rhs =
        QuantizedTensor::from_blocks(vec![32, 1], vec![Q8_0Block { scale: 2.0, values }]).unwrap();
    let result = backend.matmul_q8_0(&lhs, &rhs).unwrap();
    assert_eq!(result.as_f32().unwrap(), &[64.0]);
}

#[test]
fn computes_q4_matrix_product() {
    let backend = CpuScalar::new();
    let lhs = Tensor::from_f32(vec![1, 32], vec![1.0; 32]).unwrap();
    let rhs = QuantizedTensor4::from_blocks(
        vec![32, 1],
        vec![Q4_0Block {
            scale: 1.0,
            values: [8_u8; 32],
        }],
    )
    .unwrap();
    let result = backend.matmul_q4_0(&lhs, &rhs).unwrap();
    assert_eq!(result.as_f32().unwrap(), &[0.0]);
}

#[test]
fn computes_q4_k_matrix_product() {
    let backend = CpuScalar::new();
    let lhs = Tensor::from_f32(vec![1, 256], vec![1.0; 256]).unwrap();
    let rhs = QuantizedTensorQ4K::from_blocks(
        vec![256, 1],
        vec![Q4KBlock {
            scale: 0.1,
            min_scale: 0.0,
            scales: [63; 12],
            values: [0x99; 128],
        }],
    )
    .unwrap();
    let result = backend.matmul_q4_k(&lhs, &rhs).unwrap();
    assert!(result.as_f32().unwrap()[0] > 200.0);
}

#[test]
fn computes_q5_k_matrix_product() {
    let backend = CpuScalar::new();
    let lhs = Tensor::from_f32(vec![1, 256], vec![1.0; 256]).unwrap();
    let rhs = QuantizedTensorQ5K::from_blocks(
        vec![256, 1],
        vec![Q5KBlock {
            scale: 0.1,
            min_scale: 0.0,
            scales: [63; 12],
            high_bits: [0; 32],
            values: [0x99; 128],
        }],
    )
    .unwrap();
    let result = backend.matmul_q5_k(&lhs, &rhs).unwrap();
    assert!(result.as_f32().unwrap()[0] > 200.0);
}

#[test]
fn applies_half_split_rope_rotation() {
    let backend = CpuScalar::new();
    let input = Tensor::from_f32(vec![1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = backend
        .apply_rope(&input, &[0.0, 1.0], &[1.0, 0.0])
        .unwrap();
    assert_eq!(output.as_f32().unwrap(), &[-3.0, 2.0, 1.0, 4.0]);
}

#[test]
fn exposes_safe_cpu_isa_and_thread_configuration() {
    let backend = CpuScalar::with_config(CpuRuntimeConfig {
        isa: CpuIsa::Scalar,
        threads: 4,
        workspace_reuse: true,
    });
    assert_eq!(backend.config().threads, 4);
    assert!(backend.config().workspace_reuse);
}

#[test]
fn threaded_f32_matmul_matches_scalar_reference() {
    let lhs = Tensor::from_f32(
        vec![8, 4],
        (0..32).map(|index| index as f32 / 10.0).collect(),
    )
    .unwrap();
    let rhs = Tensor::from_f32(
        vec![4, 3],
        (0..12).map(|index| index as f32 / 7.0).collect(),
    )
    .unwrap();
    let scalar = CpuScalar::new().matmul(&lhs, &rhs).unwrap();
    let threaded = CpuScalar::with_config(CpuRuntimeConfig {
        isa: CpuIsa::Scalar,
        threads: 4,
        workspace_reuse: true,
    })
    .matmul(&lhs, &rhs)
    .unwrap();
    assert_eq!(threaded.as_f32().unwrap(), scalar.as_f32().unwrap());
}

#[test]
#[cfg(feature = "simd")]
fn f32_matmul_dispatch_matches_scalar_with_threads_and_tail_columns() {
    let lhs = Tensor::from_f32(
        vec![5, 7],
        (0..35).map(|index| index as f32 / 9.0 - 2.0).collect(),
    )
    .unwrap();
    let rhs = Tensor::from_f32(
        vec![7, 37],
        (0..259).map(|index| index as f32 / 13.0 - 8.0).collect(),
    )
    .unwrap();
    let scalar = CpuScalar::new().matmul(&lhs, &rhs).unwrap();
    for isa in [CpuIsa::Avx2, CpuIsa::Avx512, CpuIsa::Neon] {
        let actual = CpuScalar::with_config(CpuRuntimeConfig {
            isa,
            threads: 3,
            ..CpuRuntimeConfig::default()
        })
        .matmul(&lhs, &rhs)
        .unwrap();
        assert_eq!(
            actual.as_f32().unwrap(),
            scalar.as_f32().unwrap(),
            "{isa:?}"
        );
    }
}

#[test]
#[cfg(feature = "simd")]
fn quantized_output_rows_dispatch_matches_scalar_reference() {
    let lhs = Tensor::from_f32(
        vec![3, 32],
        (0..96).map(|index| index as f32 / 17.0 - 2.0).collect(),
    )
    .unwrap();
    let q8_blocks = (0..3)
        .map(|column| quantinfer_core::Q8_0Block {
            scale: 0.13 + column as f32 * 0.07,
            values: std::array::from_fn(|index| (index as i8 % 15) - 7),
        })
        .collect();
    let q8 = QuantizedTensor::from_blocks(vec![32, 3], q8_blocks).unwrap();
    let q4_blocks = (0..3)
        .map(|column| Q4_0Block {
            scale: 0.21 + column as f32 * 0.03,
            values: std::array::from_fn(|index| (index % 16) as u8),
        })
        .collect();
    let q4 = QuantizedTensor4::from_blocks(vec![32, 3], q4_blocks).unwrap();
    let scalar = CpuScalar::new();
    let expected_q8 = scalar.matmul_q8_0_output_rows(&lhs, &q8).unwrap();
    let expected_q4 = scalar.matmul_q4_0_output_rows(&lhs, &q4).unwrap();
    for isa in [CpuIsa::Avx2, CpuIsa::Avx512, CpuIsa::Neon] {
        let backend = CpuScalar::with_config(CpuRuntimeConfig {
            isa,
            ..CpuRuntimeConfig::default()
        });
        if backend.config().isa != isa {
            continue;
        }
        let actual_q8 = backend.matmul_q8_0_output_rows(&lhs, &q8).unwrap();
        let actual_q4 = backend.matmul_q4_0_output_rows(&lhs, &q4).unwrap();
        for (actual, expected) in [
            (actual_q8.as_f32().unwrap(), expected_q8.as_f32().unwrap()),
            (actual_q4.as_f32().unwrap(), expected_q4.as_f32().unwrap()),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "{isa:?}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn threaded_q8_matmul_matches_scalar_reference() {
    let lhs = Tensor::from_f32(vec![8, 32], vec![1.0; 8 * 32]).unwrap();
    let mut values = [0_i8; 32];
    values.fill(1);
    let rhs = QuantizedTensor::from_blocks(
        vec![32, 1],
        vec![quantinfer_core::Q8_0Block { scale: 0.5, values }],
    )
    .unwrap();
    let scalar = CpuScalar::new().matmul_q8_0(&lhs, &rhs).unwrap();
    let threaded = CpuScalar::with_config(CpuRuntimeConfig {
        isa: CpuIsa::Scalar,
        threads: 4,
        workspace_reuse: true,
    })
    .matmul_q8_0(&lhs, &rhs)
    .unwrap();
    assert_eq!(threaded.as_f32().unwrap(), scalar.as_f32().unwrap());
}
