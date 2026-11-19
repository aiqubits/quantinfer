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
| Metal | `quantinfer-backend-metal/native` | Optional macOS Metal adapter using the shared WGPU projection shaders and CPU fallbacks |
| WGPU | `quantinfer-backend-wgpu/native` | Optional portable GPU F32/Q4_0/Q8_0 tiled projections |
| Ascend | `quantinfer-backend-ascend/native` | Concrete CANN F32 MatMul executor: context/stream, device storage/copies, ACLNN execution and synchronization |
| T-MAC | `quantinfer-backend-tmac` | Native Rust Q4_0 output-row LUT4 with AVX2/NEON selection, scalar fallback and an optional external-kernel hook |
| KleidiAI | `quantinfer-backend-kleidiai/native` | Pinned Arm SDK Q4_0 output-row microkernel on Linux AArch64 dotprod; portable builds retain the external-kernel contract |

GPU and NPU crates are optional workspace members. They must not become
dependencies of `quantinfer-core`, the scalar CPU backend, or format readers.
When a native runtime is unavailable, resolution fails with a capability error
and callers can deliberately fall back to CPU.

T-MAC implements its LUT4 profile in Rust, with isolated unsafe SIMD intrinsics;
it does not run the upstream TVM compiler or autotuner. KleidiAI's optional
native feature compiles a fixed SDK revision through a C/unsafe FFI boundary.
Its selected microkernel also quantizes activations to Q8 with F16 scales;
this numerical profile is explicitly tested and differs from F32-activation
scalar execution. The Ascend executor isolates CANN FFI and resource ownership.
See the concrete [T-MAC](../../backends/t-mac/README.md),
[KleidiAI](../../backends/kleidiai/README.md) and
[Ascend](../../backends/ascend/README.md) profiles and acceptance boundaries.

## Verification

Run the portable checks with:

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Dedicated workflows compile optional GPU/NPU providers and exercise their
documented probe, ABI-contract, shader or device paths. T-MAC and KleidiAI
workflows execute their concrete kernels on native CPU runners. A successful
probe, mock or compile does not establish vendor-device numerical/performance
acceptance; inspect the actual steps and runner. Model conformance retains
the fixed CPU reference and explicitly selected backend checks.
