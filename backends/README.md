# Hardware backends

Backend implementations are isolated from the backend-independent crates. Vendor SDKs stay optional.

## WGPU

Enable the `wgpu` feature on `quantinfer` to use `quantinfer::wgpu::WgpuBackend`.
It runs tiled F32, Q4_0 and Q8_0 matrix multiplication through native wgpu
adapters. The quantized kernels support row-major and GGML output-row
weights. The backend uploads matrix inputs and returns a CPU tensor.
Attention, normalization, K-quant projections and the remaining
decoder operations use the scalar CPU implementation.

```rust
let backend = quantinfer::wgpu::WgpuBackend::open(false)?;
let mut registry = quantinfer::runtime::RuntimeProviderRegistry::new();
registry.register(backend.clone());
let model = quantinfer::runtime::LlamaModel::from_gguf(backend, &reader)?;
```

The provider advertises `Device::Gpu` for its accelerated operations.
Native Vulkan, Metal, and DX12 adapters depend on the host driver. Pass
`true` to `open` to select a software adapter for testing.
