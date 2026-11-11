# Backends

QuantInfer keeps device selection in `quantinfer-runtime`. A backend declares
`KernelCapability` values and is selected through `ExecutionResolver`; model
code does not instantiate a device kernel directly.

## Capability flow

1. A model or caller creates an `ExecutionRequest` containing device, dtype,
   quantization scheme, layout, operation, alignment, and optional sequence
   limits.
2. A `KernelRegistry` searches registered `RuntimeProvider` values.
3. The matching `ExecutionPlan` records the provider, kernel provider, kernel
   name, and capability used for the operation.
4. The model executes through the selected `KernelBackend` contract.

The default registry contains the safe CPU scalar provider. Applications can
register optional providers before resolving a request:

```rust
let mut registry = quantinfer_runtime::RuntimeProviderRegistry::new();
registry.register(quantinfer_runtime::CpuScalarProvider::default());
let plan = registry.resolve(&quantinfer_runtime::ExecutionRequest::cpu(
    quantinfer_core::DType::Q4_0,
    quantinfer_core::QuantizationScheme::Q4_0,
))?;
```

## Available providers

| Provider | Build gate | Current scope |
| --- | --- | --- |
| CPU scalar | default | F32, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K; reference attention and KV cache |
| CPU SIMD | `quantinfer-backend-cpu/simd` | Runtime checked AVX2, AVX-512 and AArch64 NEON scaled-add kernels; scalar fallback remains the reference |
| CUDA | `quantinfer-backend-cuda/native` | Optional dynamic CUDA driver/NVRTC projection backend with capability checks |
| Metal | `quantinfer-backend-metal/native` | Optional macOS MSL projection backend and CPU fallback conformance |
| WGPU | `quantinfer-backend-wgpu/native` | Optional portable GPU F32/Q4_0/Q8_0 tiled projections |
| Ascend | `quantinfer-backend-ascend/native` | Optional CANN ACL probe and ACLNN MatMul workspace/execute plan |
| T-MAC | `quantinfer-backend-tmac` | Explicit CPU Q4_0 capability contract for a linked LUT kernel |
| KleidiAI | `quantinfer-backend-kleidiai` | Explicit Arm CPU Q8_0 capability contract for a linked micro-kernel |

GPU and NPU crates are optional workspace members. They must not become
dependencies of `quantinfer-core`, the scalar CPU backend, or format readers.
When a native runtime is unavailable, resolution fails with a capability error
and callers can deliberately fall back to CPU.

The T-MAC and KleidiAI crates intentionally contain no vendor library or
unsafe FFI. They provide the stable capability boundary and target detection;
an application links its chosen upstream kernel and registers the provider
only when that execution implementation is present.

## Verification

Run the portable checks with:

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

The CUDA, Metal, WGPU and Ascend workflows compile their optional providers;
native hardware jobs run only when a configured runner is available. The
model conformance workflow compares the selected backend against the pinned
CPU reference before accepting a change.
