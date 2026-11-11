# Hardware backends

Backend implementations are isolated from the backend-independent crates. Vendor SDKs stay optional.

## WGPU

Enable the `wgpu` feature on `quantinfer` to use `quantinfer::wgpu::WgpuBackend`.
It runs tiled F32, Q4_0 and Q8_0 matrix multiplication through native wgpu
adapters. The quantized kernels support row-major and GGML output-row
weights. The backend uploads matrix inputs and returns a CPU tensor.
Wide F32 projections are split into column panels when required by the
adapter's buffer limit.
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

Build the CLI with `--features wgpu` and select the backend for generation.

```sh
cargo run --release -p quantinfer-cli --features wgpu -- generate --backend wgpu \
  --input model.gguf --prompt "The capital of France is" --max-new-tokens 8 --max-sequence-length 64
```

Standalone GGUF and QIF files can use WGPU directly. If a model package
declares execution variants, it must include one for WGPU. Inference
profiles and memory budgets supplied through the CLI apply to CPU execution.

## CUDA

Enable the `cuda` feature on `quantinfer` to use `quantinfer::cuda::CudaBackend`
on Linux or Windows. `CudaBackend::open(0)` selects the first visible NVIDIA
device. The backend requires compute capability 7.5 or higher, a compatible
driver and the CUDA 12.8 NVRTC library on the system library path.

CUDA accelerates F32, Q4_0 and Q8_0 projections. Packed weights keep their
codes and scales in either row-major or GGML output-row order. Activations
are uploaded for each projection and the result is returned as a CPU tensor.
The remaining decoder operators use the scalar CPU implementation.

```rust
let backend = quantinfer::cuda::CudaBackend::open(0)?;
let mut registry = quantinfer::runtime::RuntimeProviderRegistry::new();
registry.register(backend.clone());
let model = quantinfer::runtime::Qwen3Model::from_gguf(backend, &reader)?;
```

Build the CLI with `--features cuda` and use `generate --backend cuda`.
Device zero follows the host's `CUDA_VISIBLE_DEVICES` selection. Packages
that declare execution variants must include one for the CUDA provider.

The hosted CUDA workflow compiles the projection kernel with NVRTC. Run the
numerical comparison on a CUDA machine with `cargo test --release --locked
-p quantinfer-backend-cuda --features native --test native -- --ignored`.
The same test can run through the CUDA workflow using a self-hosted runner
with the label configured in `QUANTINFER_CUDA_RUNNER`.

## Metal

On macOS, enable the `metal` feature on `quantinfer` to use
`quantinfer::metal::MetalBackend`. It selects only the system Metal adapter
through WGPU, while exposing an independent `metal` runtime provider name.
The provider accelerates the same F32, Q4_0 and Q8_0 projection operations
and keeps the scalar CPU implementation for the remaining decoder operators.

```sh
cargo run --release --locked -p quantinfer-cli --features metal -- generate \
  --backend metal --input model.gguf --prompt "The capital of France is" \
  --max-new-tokens 8 --max-sequence-length 64
```

## Ascend CANN

The optional Ascend crate now probes the CANN ACL runtime without adding a
vendor dependency to the workspace. Set `ASCEND_ACL_LIBRARY` to the ACL
shared library path and call `AscendRuntime::open()` to initialize CANN and
query visible devices. Matrix operators still require the CANN operator ABI
integration before an Ascend execution provider can be selected.
`load_float_matmul_abi()` checks the ACLNN two-phase MatMul symbols in the
operator library configured by `ASCEND_OP_LIBRARY`.
The runtime capability model exposes Ascend devices as `Device::Npu`.
After the ABI check, `AscendRuntime::execution_provider()` supplies the
F32 MatMul plan to the shared runtime registry. Tensor creation and operator
submission remain the next device-side step.
