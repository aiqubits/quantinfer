# RFCs

This directory records design decisions that affect public contracts. An RFC
should explain the problem, invariants, compatibility impact, migration path,
and the tests that make the decision observable.

## Current decisions

### Semantic model boundary

Model runtime code consumes `ModelSpec` and semantic tensor roles. File names,
SafeTensors keys, and GGUF metadata are translated by format adapters. This
keeps a second dense model from inheriting Qwen3-specific assumptions.

### Descriptor-first quantization

`TensorStorageDescriptor` and `QuantizedTensorDescriptor` describe logical
shape, physical layout, block/group size, scales, zero points, packing, and
backend requirements. A layout transform is explicit and never silently
performs a lossy second quantization.

### Capability-selected execution

`RuntimeProvider`, `KernelCapability`, `KernelRegistry`, and
`ExecutionPlan` are the only device-selection boundary. Optional CUDA, Metal,
WGPU, Ascend, and future CPU/SoC providers stay outside the core dependency
graph and can be absent from a portable build.

### Model packages and QIF

Manifests and checksum-verified packages are the stable deployment boundary.
QIF is an optional container and is not required by the runtime; GGUF and
legacy CLI loading remain compatible entry points.

### Compatibility and rejection

Unsupported architecture fields, layouts, dtypes, and quantization methods
return actionable errors. A change that broadens support must add a fixture and
conformance coverage rather than weakening an existing rejection.

Future proposals should add a dated RFC file when they change any of these
boundaries. Routine implementation notes belong in the relevant package or
user guide instead.
