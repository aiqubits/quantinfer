# KleidiAI backend

The optional `native` feature compiles a real NEON dotprod microkernel from
[Arm KleidiAI](https://github.com/ARM-software/kleidiai) at the exact revision in
[`vendor-revision`](vendor-revision). The source must be a clean Git checkout.
The current native profile is little-endian Linux AArch64 with runtime dotprod
support. Portable builds retain the caller-supplied kernel contract.

```sh
git clone https://github.com/ARM-software/kleidiai.git /tmp/kleidiai
git -C /tmp/kleidiai checkout 9b65dd111ab5cc8a6c0d3c9c0883b69ad97eeab5
export QUANTINFER_KLEIDIAI_SOURCE_DIR=/tmp/kleidiai
cargo test --release --locked -p quantinfer-backend-kleidiai --features native -- --nocapture
cargo run --release --locked -p quantinfer-cli --features kleidiai -- generate \
  --backend kleidiai --input model.qif --tokens 1,2,3 --max-new-tokens 8
```

`KleidiAiBackend::open()` selects the concrete `NativeKleidiAiKernel`. Its native
operation is `matmul_q4_0_output_rows`: the RHS has logical shape `[K,N]` and
physical blocks grouped by output column, with K divisible by 32. Row-major
Q4_0 and other operators use the existing CPU fallback. The capability list
includes those fallback operations; backend identity does not mean every op
executes inside the SDK.

Each weight block's 32 codes and F16 scale are repacked without changing values.
Finite Core scales that cannot be represented exactly in F16 are rejected on
the SDK path rather than rounded. This matches GGML Q4_0 and the current QIF
encoding, whose stored scales are already F16. Bit packing preserves the GGML
`index 0 / index 16` nibble order; the SDK performs its own signed/interleaved
packing. Other layouts are not reinterpreted as output rows.

The chosen kernel family is `qsi8d32p × qsi4c32p`: it **also quantizes F32
activations to signed Q8 in groups of 32 with F16 scales**. It therefore has a
different numerical profile from the CPU scalar path with F32 activations.
This is an explicit optional backend, not a lossless execution claim. Native
tests check exact-grid activations and a bound on fractional activation
quantization error, including negative/zero weight scales and tail tiles.
Nonfinite activations and activation scales that would underflow/overflow the
F16 profile are rejected before the SDK's float-to-integer conversions.

The C bridge includes upstream headers so the compiler checks ABI signatures.
The compiled SDK symbols are namespaced to prevent another adapter's SDK copy
from substituting an incompatible packed-layout ABI.
Rust validates products, tile padding and SDK buffer sizes and allocates
64-byte-aligned, initialized, nonoverlapping workspaces. Required ISA support is
checked before exposing the native kernel. Weight packing is currently done
per call; no cross-model speedup or optimal cache strategy is claimed.

Validation boundaries:

- Portable tests check lossless weight packing, invalid scales/codes and numeric
  resource bounds. Scalar hooks remain separately tested.
- Five matrix shapes, each with exact-grid and fractional activations, exercise
  the real upstream assembler under the local AArch64 QEMU runner. This is
  instruction-level emulation, not hardware performance evidence.
- The dedicated [workflow](../../.github/workflows/kleidiai.yml) runs the real SDK
  tests on a native ARM runner, requires the explicit verification marker, and
  runs CLI tests and native Clippy. The initial [native ARM run](https://github.com/aiqubits/quantinfer/actions/runs/44236519435/job/134573867442)
  passed the SDK tests and CLI integration checks, including the subprocess
  checks which exited 127 under the local emulator. The emulator failure was
  not recorded as a local pass.
- Full model quality and device-specific throughput/energy profiling are
  separate from this kernel contract matrix.

The selected upstream
[kernel](https://github.com/ARM-software/kleidiai/blob/9b65dd111ab5cc8a6c0d3c9c0883b69ad97eeab5/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c),
[activation packer](https://github.com/ARM-software/kleidiai/blob/9b65dd111ab5cc8a6c0d3c9c0883b69ad97eeab5/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c)
and [weight packer](https://github.com/ARM-software/kleidiai/blob/9b65dd111ab5cc8a6c0d3c9c0883b69ad97eeab5/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c)
define the concrete ABI and computation used here. The similarly named BF16-scale
packer is a different profile and is not substituted for this F16 path.
