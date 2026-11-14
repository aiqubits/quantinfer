#![cfg(all(feature = "native", not(target_arch = "wasm32")))]

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_backend_wgpu::WgpuBackend;
use quantinfer_core::{
    DType, Device, Q4_0Block, Q8_0Block, QuantizationScheme, QuantizedTensor, QuantizedTensor4,
    Tensor, OUTPUT_ROWS_LAYOUT,
};
use quantinfer_format::PackedTensorReader;
use quantinfer_kernel::KernelBackend;
use quantinfer_runtime::{
    select_token, ExecutionRequest, GenerationConfig, KvCache, LlamaModel, Operation,
    RuntimeProviderRegistry,
};

fn open() -> WgpuBackend {
    let software = std::env::var_os("QUANTINFER_WGPU_SOFTWARE").is_some();
    let backend = WgpuBackend::open(software).expect("native wgpu adapter");
    let info = backend.adapter_info();
    println!(
        "adapter={:?} device={:?} name={}",
        info.backend, info.device_type, info.name
    );
    backend
}

fn close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) -> f32 {
    assert_eq!(lhs.shape(), rhs.shape());
    let maximum = lhs
        .as_f32()
        .unwrap()
        .iter()
        .zip(rhs.as_f32().unwrap())
        .map(|(lhs, rhs)| {
            assert!(lhs.is_finite() && rhs.is_finite());
            (lhs - rhs).abs()
        })
        .fold(0.0, f32::max);
    assert!(
        maximum <= tolerance,
        "maximum difference {maximum} exceeds {tolerance}"
    );
    maximum
}

fn matrix(rows: usize, cols: usize) -> Tensor {
    let values = (0..rows * cols)
        .map(|index| ((index * 19 + index / 7) % 41) as f32 * 0.125 - 2.5)
        .collect();
    Tensor::from_f32(vec![rows, cols], values).unwrap()
}

#[test]
#[ignore = "requires a native WGPU adapter"]
fn matches_cpu_for_partial_tiles_and_packed_weights() {
    let gpu = open();
    let cpu = CpuScalar::new();
    let lhs = matrix(5, 35);
    let rhs = matrix(35, 17);
    close(
        &gpu.matmul(&lhs, &rhs).unwrap(),
        &cpu.matmul(&lhs, &rhs).unwrap(),
        1e-5,
    );
    assert!(gpu.matmul(&lhs, &matrix(34, 17)).is_err());

    let lhs = matrix(5, 64);
    let shape = vec![64, 17];
    let q4 = QuantizedTensor4::from_blocks(
        shape.clone(),
        (0..34)
            .map(|block| Q4_0Block {
                scale: (block % 7) as f32 * 0.125 - 0.375,
                values: std::array::from_fn(|index| ((index * 5 + block) % 16) as u8),
            })
            .collect(),
    )
    .unwrap();
    let q8 = QuantizedTensor::from_blocks(
        shape,
        (0..34)
            .map(|block| Q8_0Block {
                scale: (block % 7 + 1) as f32 * 0.03125,
                values: std::array::from_fn(|index| (index * 11 + block * 17) as i8),
            })
            .collect(),
    )
    .unwrap();
    close(
        &gpu.matmul_q4_0(&lhs, &q4).unwrap(),
        &cpu.matmul_q4_0(&lhs, &q4).unwrap(),
        1e-5,
    );
    close(
        &gpu.matmul_q8_0(&lhs, &q8).unwrap(),
        &cpu.matmul_q8_0(&lhs, &q8).unwrap(),
        1e-5,
    );
    close(
        &gpu.matmul_q4_0_output_rows(&lhs, &q4).unwrap(),
        &cpu.matmul_q4_0_output_rows(&lhs, &q4).unwrap(),
        1e-5,
    );
    close(
        &gpu.matmul_q8_0_output_rows(&lhs, &q8).unwrap(),
        &cpu.matmul_q8_0_output_rows(&lhs, &q8).unwrap(),
        1e-5,
    );

    let mut registry = RuntimeProviderRegistry::new();
    registry.register(gpu);
    let mut request = ExecutionRequest {
        device: Device::Gpu,
        tensor_layout: OUTPUT_ROWS_LAYOUT.to_owned(),
        operation: Some(Operation::QuantizedMatmul),
        alignment: Some(32),
        ..ExecutionRequest::cpu(DType::Q4_0, QuantizationScheme::Q4_0)
    };
    assert_eq!(registry.resolve(&request).unwrap().provider_name, "wgpu");
    request.operation = Some(Operation::Attention);
    assert!(registry.resolve(&request).is_err());
}

#[test]
#[ignore = "requires a native WGPU adapter and QUANTINFER_WGPU_GGUF"]
fn matches_smollm2_logits_and_generation() {
    let gpu = open();
    let path = std::env::var_os("QUANTINFER_WGPU_GGUF").expect("pinned SmolLM2 GGUF path");
    let reader = PackedTensorReader::open_artifact(std::fs::read(path).unwrap()).unwrap();
    let model = LlamaModel::from_gguf(gpu, &reader).unwrap();
    let reference = LlamaModel::from_gguf(CpuScalar::new(), &reader).unwrap();
    drop(reader);
    let cache = || {
        KvCache::new(
            model.config.num_hidden_layers,
            64,
            model.config.num_key_value_heads(),
            model.config.effective_head_dim(),
        )
        .unwrap()
    };
    let mut gpu_cache = cache();
    let mut cpu_cache = cache();
    let mut tokens = vec![504, 3575, 282, 4649, 314];
    let mut generated = Vec::new();
    let mut maximum = 0.0_f32;
    let config = GenerationConfig::default();
    for _ in 0..3 {
        let gpu = model.forward(&tokens, &mut gpu_cache).unwrap();
        let cpu = reference.forward(&tokens, &mut cpu_cache).unwrap();
        maximum = maximum.max(close(&gpu, &cpu, 0.002));
        let vocab = model.config.vocab_size;
        let last_row = |tensor: &Tensor| {
            let values = tensor.as_f32().unwrap();
            Tensor::from_f32(vec![vocab], values[values.len() - vocab..].to_vec()).unwrap()
        };
        let next = select_token(&last_row(&gpu), &config).unwrap();
        assert_eq!(next, select_token(&last_row(&cpu), &config).unwrap());
        generated.push(next);
        tokens = vec![next];
    }
    println!("SmolLM2 maximum_logit_difference={maximum:.8} generated_tokens={generated:?}");
}
