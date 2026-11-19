# AutoRound

`AutoRoundProvider` calibrates a floating `[input, output]` linear matrix using
the supplied activation rows. It optimizes the SignRound projection loss
`mean((X (W_quantized - W))²)` across each output's entire input vector.
Output rows are independent and may be scheduled across CPU workers.

The provider learns a rounding offset per weight and a symmetric clipping
multiplier per group. It uses straight-through rounding and signed updates.
Offsets stay in `[-0.5, 0.5]` and clipping multipliers in `[0, 1]`. The
forward loss uses the snapped F16 scale that will be stored and includes the
initial round-to-nearest grid in the selection. Groups are multiples of 32
and can have a shorter final group. The chosen codes and scales are written
directly as native Q4_0 or Q8_0 output rows.

This profile follows the reconstruction and signed-rounding approach in the
[AutoRound paper](https://arxiv.org/abs/2309.05516) and the upstream
[per-layer SignRound workflow](https://github.com/intel/auto-round/blob/933b5af408109110e140d17131d348851e202199/auto_round/algorithms/sign_round/quantizer.py).
The supported profile uses symmetric F16 grids and per-projection loss.
Upstream asymmetric zero points, transformer-block backpropagation and
vendor-specific kernels require different representations and execution paths.
The existing checkpoint importer continues to read GPTQ-compatible exports.

Independent Torch autograd fixtures cover both bit widths, two group sizes,
and a shorter final group. Rust checks the loss, every stored scale and every
code against that reference and compares one worker with multiple workers.
The [whole-model CLI](../../tools/quantinfer-convert/README.md#autoround)
propagates the packed layers, verifies the artifact and checks complete
logits, generation and KV parity on reload. The fixed SmolLM2 test optimized
210 projections and passed its four validation prompts at KL ≤ 0.5 and
Top-10 overlap ≥ 6. This is the measured profile and prompt set.
