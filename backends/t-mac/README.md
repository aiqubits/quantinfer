# Native LUT4 provider

`NativeLut4Kernel` implements the bit-plane/table-lookup method described by
[T-MAC](https://github.com/microsoft/T-MAC) in Rust. The concrete profile is
Q4_0 output-row weights with F32 activations/scales. It includes AVX2 gather,
AArch64 NEON byte-table lookup and a scalar LUT fallback. This implementation
does not invoke Microsoft's TVM compiler, generated binaries or autotuner, and
does not claim their benchmark results. The existing `TmacKernel` hook remains
available for integrating downstream/upstream-generated kernels.

```sh
cargo run --release --locked -p quantinfer-cli --features tmac -- generate \
  --backend t-mac --input model.qif --tokens 1,2,3 --max-new-tokens 8
cargo test --release --locked -p quantinfer-backend-tmac -- --nocapture
```

The native operation is `matmul_q4_0_output_rows`, with logical RHS shape `[K,N]`,
K divisible by 32, and one scale per 32 input values per output column.
`TmacBackend::open()` chooses the supported lookup ISA and exposes the actual
choice through `implementation()`. Other operators and the row-major Q4_0
method retain the CPU fallback, also reflected in the capability list.

For each four activations, construct a 16-entry subset-sum table `L`. Convert
offset-binary Q4_0 codes to signed two's-complement planes with `code ^ 8`.
Each plane supplies a four-bit lookup mask `p0..p3`; its contribution is:

```text
L[p0] + 2*L[p1] + 4*L[p2] - 8*L[p3]
```

Eight such groups share a Q4_0 block scale. The kernel consumes packed planes
without expanding a floating-point weight matrix. It preserves the original
F32 weight scales and does not introduce activation Q8/F16 quantization;
floating-point reassociation still permits small rounding differences from
the scalar multiply-accumulate order.

`Lut4Weights::from_output_rows` builds an immutable packed snapshot. Applications
may reuse it with `NativeLut4Kernel::matmul_prepared`; subsequent mutations of
the source Core tensor do not change that snapshot. Its plane/scale payload
uses `K*N/2 + 4*K*N/32` bytes, excluding allocation/struct overhead. The ordinary
`KernelBackend` call prepares a fresh snapshot each time. Lookup tables are
allocated once per call and reused for each input row. These choices do not
establish an end-to-end speedup; performance must be measured for the workload.

The lookup module contains the SIMD unsafe boundary. Selection checks the ISA,
and every mask is within the 16-element table. Shape/size arithmetic, scale/code
validity, finite inputs and arithmetic overflow are checked before returning a
tensor. Nonfinite inputs or table/output overflow return an error.

Validation includes every one of the 65,536 possible packed plane words,
independent F64 matrix references for several shapes, scalar/SIMD comparisons,
negative/zero/non-F16 scales, immutable prepared snapshots and rejected invalid
profiles. Local x86 execution and AArch64 QEMU execution pass. The dedicated
[workflow](../../.github/workflows/tmac.yml) requires actual AVX2/NEON markers on
native x86/ARM runners and runs CLI tests and Clippy. Generic kernel tests are
separate from full-model quality, throughput and energy measurements.
