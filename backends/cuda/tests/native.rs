#![cfg(all(feature = "native", any(target_os = "linux", target_os = "windows")))]

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_backend_cuda::CudaBackend;
use quantinfer_core::{DType, Device, QuantizationScheme, Tensor, OUTPUT_ROWS_LAYOUT};
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{quantize_q4_0, quantize_q8_0};
use quantinfer_runtime::{ExecutionRequest, Operation, RuntimeProviderRegistry};
use quantinfer_test_support::deterministic_tensor;

#[test]
#[ignore = "requires a CUDA device and NVRTC"]
fn matches_cpu_for_dense_and_packed_projections() {
    let gpu = CudaBackend::open(0).unwrap();
    println!("device={}", gpu.device_name());
    let cpu = CpuScalar::new();
    let left = deterministic_tensor(vec![19, 64], 1);
    let right = deterministic_tensor(vec![64, 35], 2);
    let q4 = quantize_q4_0(&right).unwrap();
    let q8 = quantize_q8_0(&right).unwrap();
    close(
        gpu.matmul(&left, &right).unwrap(),
        cpu.matmul(&left, &right).unwrap(),
    );
    close(
        gpu.matmul_q4_0(&left, &q4).unwrap(),
        cpu.matmul_q4_0(&left, &q4).unwrap(),
    );
    close(
        gpu.matmul_q8_0(&left, &q8).unwrap(),
        cpu.matmul_q8_0(&left, &q8).unwrap(),
    );
    close(
        gpu.matmul_q4_0_output_rows(&left, &q4).unwrap(),
        cpu.matmul_q4_0_output_rows(&left, &q4).unwrap(),
    );
    close(
        gpu.matmul_q8_0_output_rows(&left, &q8).unwrap(),
        cpu.matmul_q8_0_output_rows(&left, &q8).unwrap(),
    );
    let partial_inner = deterministic_tensor(vec![19, 37], 3);
    let right = deterministic_tensor(vec![37, 35], 4);
    close(
        gpu.matmul(&partial_inner, &right).unwrap(),
        cpu.matmul(&partial_inner, &right).unwrap(),
    );
    assert!(gpu.matmul(&left, &right).is_err());

    let mut registry = RuntimeProviderRegistry::new();
    registry.register(gpu);
    let mut request = ExecutionRequest {
        device: Device::Gpu,
        tensor_layout: OUTPUT_ROWS_LAYOUT.to_owned(),
        operation: Some(Operation::QuantizedMatmul),
        alignment: Some(32),
        ..ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0)
    };
    assert_eq!(registry.resolve(&request).unwrap().provider_name, "cuda");
    request.operation = Some(Operation::Attention);
    assert!(registry.resolve(&request).is_err());
}

fn close(actual: Tensor, expected: Tensor) {
    assert_eq!(actual.shape(), expected.shape());
    for (&actual, &expected) in actual
        .as_f32()
        .unwrap()
        .iter()
        .zip(expected.as_f32().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}
