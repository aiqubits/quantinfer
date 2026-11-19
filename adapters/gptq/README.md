# GPTQ adapter and calibrated provider

This crate owns two separate capabilities: importing existing GPTQ checkpoint
tensors, and producing native signed quantized matrices with full-Hessian GPTQ.
Both are safe Rust. Neither requires PyTorch, Python, CUDA or GGML at runtime.

## Calibrated production profile

`GptqProvider` implements `QuantizationProvider` and
`CalibratedQuantizationProvider<Calibration = HessianCalibration>`. Its input
is a finite F32 linear matrix `[input, output]`. Calibration observes the
actual `[token_rows, input]` activations before that projection. It accumulates
the full Gram matrix with F64 arithmetic and uses `H = 2 XᵀX / token_rows`.
Different prompt lengths contribute their actual numbers of token rows.

The algorithm handles dead channels, adds positive relative diagonal damping,
computes `chol(H⁻¹)` in upper-triangular order, and propagates each quantized
channel's error into all remaining channels. It does not replace the Hessian
with its diagonal. An uncalibrated `quantize()` call fails explicitly.

The profile uses static contiguous signed groups, four/eight bits, F16 scales,
and optional activation-order processing. Static groups retain their original
input coordinates after the processing permutation. The input width and group
size are multiples of 32; a final shorter group is allowed. Scales follow the
upstream symmetric min/max profile and are rounded to F16 before quantization
and feedback. Underflow uses the smallest positive F16 scale; overflow fails.
Rounding uses ties-to-even. Codes and scales map directly to output-row
Q4_0/Q8_0 blocks, without a second quantization or a new QIF encoding ID.

```rust
use quantinfer_adapter_gptq::{GptqProvider, HessianCalibration};
use quantinfer_quant::{CalibratedQuantizationProvider, QuantizationProvider};

# fn example(weights: &quantinfer_core::Tensor, activations: &quantinfer_core::Tensor)
# -> quantinfer_quant::Result<()> {
let channels = weights.shape().dims()[0];
let mut calibration = HessianCalibration::new(channels, 128 * 1024 * 1024)?;
calibration.observe_rows(activations)?;
let provider = GptqProvider::default();
let packed = provider.quantize_calibrated(weights, &calibration)?;
let descriptor = provider.storage_descriptor(weights.shape().clone())?;
let error = provider.error_report(weights, &packed)?;
# let _ = (descriptor, error);
# Ok(()) }
```

Use this provider's descriptor and dequantizer with its output-row blocks.
Treating those bytes as a row-major `[input, output]` matrix changes the linear
function. QIF writers canonicalize the internal output-row layout name.

`HessianCalibration::new` bounds the statistics allocation. Provider options
separately bound the algorithm-owned Hessian/working-weight/result buffers;
caller-owned inputs and statistics are excluded. Allocation failure and an
insufficient budget return errors. These are workspace checks, not a process
RSS guarantee. The dense factorization has cubic cost in the input width;
this implementation does not claim a tuned BLAS/GPU quantization performance.

## Independent and real-model validation

The algorithm follows the full-Hessian feedback in pinned upstream
[`gptq.py`](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py).
The fixed profile's symmetric grid derives from
[`quant.py`](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py),
with F16 scale freezing made explicit. It is a static-group profile, not a
claim to implement every GPTQ SDK option or external checkpoint convention.

Eight fixed vectors cover four/eight bits, 32/64 groups and both activation
orders. Rust compares every code and scale with the original upstream Torch
implementation. The independent reference script verifies both source hashes,
executes upstream feedback on CPU, and bypasses only CUDA synchronization.
CI recomputes vectors from their literal input arrays. QIF publishes all
eight cases with none/zstd/LZ4 and checks exact blocks plus native CPU results.

Additional tests cover full off-diagonal statistics, split-batch equivalence,
the inverse-factor matrix identity, invalid inputs, missing/zero calibration,
workspace limits, dead channels, F16 subnormals, ties and partial groups.

The ignored real-model test calibrates the first SmolLM2-135M query projection
on three declared prompts (21 token rows), produces Q4_0 and runs the complete
model through SINGLE/none and PACKAGE/zstd. It checks the same-grid RTN
projection baseline, holdout full-vocabulary KL (`<0.05`) and Top-10 overlap
(`>=8`), generation repeatability, KV and unchanged packed blocks. The source
F32 model also checks fixed Transformers logits/token vectors. This is one
quantized real projection in a mixed model, not an all-layer GPTQ quality gate.

```sh
QUANTINFER_GPTQ_SOURCE_GGUF=/path/to/verified-smollm2-f32.gguf \
QUANTINFER_GPTQ_SOURCE_SHA256=EXPECTED_SHA256 \
QUANTINFER_GPTQ_REPORT=/path/to/quality.json \
cargo test --release --locked -p quantinfer-adapter-gptq \
  --test model_conformance -- --ignored --nocapture
```

Whole-model quantization/calibration orchestration and Studio integration
remain to be implemented. The API can quantize compatible individual matrices;
the single-projection test does not establish full-model GPTQ acceptance.

## Existing checkpoint import

`dequantize` and `dequantize_with_zero_point_layout` retain their existing
2/4/8-bit GPTQ SafeTensors support, including explicit zero-point orientations,
modular v1 zero restoration, optional `g_idx` and sharded sources. Their result
is a validated F32 tensor. Native calibrated production currently supports the
signed four/eight-bit profile above; external asymmetric/two-bit imports do
not imply a native execution kernel for those checkpoint formats.
