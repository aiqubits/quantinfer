# Ascend backend

The optional `native` feature provides an actual CANN/ACLNN F32 MatMul executor,
`CannMatmulExecutor`, and `AscendBackend::open(device)`. Portable default builds
do not load CANN. A downstream `AscendMatmulExecutor` remains an extension point.

The concrete executor accepts nonempty, compatible rank-2 F32 Core tensors. It
creates an explicit device context/stream for each call, copies inputs, creates
dense ND ACL tensors, queries workspace, invokes `aclnnMm`, synchronizes, and
copies the output to a Core tensor. Non-MatMul operators retain the documented
CPU fallback. Quantized matmul is not advertised as an NPU capability.

F32 `KEEP_DTYPE` (`cubeMathType=0`) is requested. Devices/SDK combinations that
cannot execute that mode return an error; the wrapper does not silently choose
FP16/HF32. This is a functional binding, not a claim that repeated host/device
transfers or a per-call context deliver optimal inference performance.

```sh
cargo run --locked -p quantinfer-cli --features ascend -- generate \
  --backend ascend --input model.qif --tokens 1,2,3 --max-new-tokens 8
```

This CLI path requires Linux and a working SDK/driver/device. CANN's libraries
must be discoverable by the loader, or selected with `ASCEND_ACL_LIBRARY` and
`ASCEND_OP_LIBRARY` (default `libascendcl.so` and `libopapi.so`). Empty overrides
use defaults. Library and operator dependencies must come from a trusted,
compatible CANN installation.

`AscendRuntime::open_library` / `load_float_matmul_library` also accept explicit
paths. The process can have one managed ACL runtime at a time; share an
`Arc<AscendRuntime>` across executors. This avoids one handle's process-global
`aclFinalize` invalidating another. Each call restores a previously active
explicit context. Calls on one executor are serialized; no raw context handle
is retained across calling threads.

The executor marks the ACL operator reusable immediately after workspace
planning so it can explicitly destroy it after synchronization, including when
workspace allocation fails. It uses a fresh plan for each matmul. Partial
allocation/tensor/context/stream/copy failures release acquired resources.
If completion or operator ownership becomes uncertain, the runtime retains the
affected resources and library lifetime, marks itself quarantined, and rejects
further executor creation/execution. A process restart is required in that
case. This prevents releasing storage an NPU task may still use.

The lower-level `prepare_float_matmul` and one-shot plan `execute` methods are
unsafe; their callers own pointer validity, device/shape compatibility and all
asynchronous lifetimes. Applications can use the Core-tensor executor to avoid
handling raw ACL pointers themselves.

Validation is deliberately separated:

- `cargo test -p quantinfer-backend-ascend --features native` compiles an
  independent C ABI double, checks an asymmetric matrix result, moves execution
  to another thread, checks call counts/context restoration, and injects failures
  in allocation, tensor construction, workspace, execution, synchronization,
  context/stream creation and copies. Quarantine tests run in child processes.
  Destruction and context-restore failures are also injected and must not be
  reported as successful operations or trigger repeated destruction.
  These are contract tests; the C double is not an NPU.
- Two compile-fail doctests enforce raw-pointer unsafe call boundaries. Native
  Clippy and CLI accelerator feature compilation are CI gates.
- `cargo test -p quantinfer-backend-ascend --features native -- --ignored`
  explicitly executes a small real-device F32 comparison. It requires a
  configured SDK/driver/device and is not claimed as passed by portable CI.
  Full model accuracy, device throughput and peak-device-memory measurement
  remain separate hardware acceptance work.

The ABI and ownership follow CANN's official
[MatMul API/example](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnMm.md),
[executor reuse](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00041.html),
[executor destruction](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/aolapi/operatorlist_00021.html),
[initialization/finalization](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0005.html)
and [context error codes](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendgraphapi/aclpythondevg_01_0911.html).
