# Mandatory runtime requirements

Model requirements are conjunctive: every name must be understood and satisfied.
Format readers and `inspect` retain declarations without selecting a runtime.
Runtime validation first checks SemVer, registered names and kernel operation
spellings. Execution then matches backend declarations against the selected
implementation. Unknown names fail; optional metadata hints do not register
mandatory capabilities.

`quantinfer_runtime::RUNTIME_FEATURES` lists the format semantics included in
every current runtime build:

| `compatibility.required_features` name | Supported semantics |
| --- | --- |
| `qif.metadata.map` | Preserve typed QMB maps, including nested maps |
| `qif.tensor.alias` | Resolve native QIF physical tensor aliases |
| `qif.codec.zstd` | Decode the current independent zstd chunk profile |
| `qif.codec.lz4` | Decode the current length-prefixed LZ4 chunk profile |

These names do not promise filesystem access, remote transport, device support,
or that all tensors remain resident. They refer to parsing/decoding capability;
actual control/payload validation still applies.

`quantinfer_kernel::BackendId` registers the exact, case-sensitive tensor
`descriptor.backend_requirements` names: `cpu`, `wgpu`, `cuda`, `metal`, `ascend`,
`t-mac`, `kleidiai`. Each means that the corresponding kernel implementation
must be selected for execution. CPU's optional SIMD dispatch retains identity
`cpu`. A list containing two different identities cannot be satisfied by one
selected backend; it does not mean an OR choice or an offload plan.

Backend identity does not attest that every operation runs on that device.
Some implementations have CPU fallbacks. Kernel capability, dtype/layout,
alignment, context/head limits and operator checks remain separate. Registering
Ascend/T-MAC/KleidiAI names does not supply their missing vendor kernels or
complete hardware acceptance.

`KernelBackend::backend_id()` defaults to `None` for custom implementations.
Without an explicit registered identity they can execute models with no
mandatory backend declaration, but cannot satisfy one. New mandatory names need
an explicit registry extension and matching implementation/tests; merely putting
a string in a manifest does not grant support.

`validate_manifest_for_runtime` validates names and minimum version only;
`validate_manifest_for_backend` and `validate_manifest_for_backend_id` also check
the selected backend. CPU resolution, GPU planning, `DenseModel::with_backend`
and direct Qwen3/Llama artifact constructors apply the execution check before
loading tensor payloads. Model-spec inspection can remain backend independent.
Raw tensor/model constructors have no artifact manifest to enforce.

Tests exercise all registered identities against every other identity, unknown
features, unregistered custom identity, impossible conjunctions, both direct
Dense loaders, CPU inference with positive requirements, and GPU capability
planning without opening a device. Device correctness/performance remains a
separate conformance matrix.
