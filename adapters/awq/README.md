# AWQ

The adapter reads AutoAWQ packed checkpoints and also exposes the native
`AwqProvider` for calibration and weight production. See the
[converter workflow](../../tools/quantinfer-convert/README.md#awq) for
model-level calibration, artifact publication and quality validation.

The provider searches channel scales from mean absolute calibration
activations. For each ratio it evaluates the active model operation with
simulated quantized weights and selects the minimum reconstruction loss.
The converter folds the winning scales into adjacent projections and
normalizations, with compensation that preserves the mathematical function
before quantization. The clipping search evaluates ten shrink factors
from 1.00 to 0.55 against token-output error for each input group and
output channel. The Gram matrix retains the within-group correlations.

The signed QIF profile freezes each positive group scale to F16 before
rounding, with a nearest-even signed code grid. A group can span multiple
32-value Q4_0 or Q8_0 storage blocks. This profile uses the search and
compensation described by [AWQ](https://arxiv.org/abs/2306.00978) and the
pinned [scaling code](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)
and [clipping code](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py).
The native profile uses symmetric signed F16 grids and supports both
four-bit and eight-bit output.

Four [Torch vectors](../../tests/golden/awq-signed-profile.json) cover both
bit widths and two group sizes, including a final partial group. The
[independent reference](../../tests/conformance/awq_reference.py) directly
computes the token outputs for each candidate. Tests check ratio, channel
scales, clipping bounds, losses and encoded codes/scales. The whole-model
acceptance case uses SmolLM2 and compares the published SINGLE and PACKAGE
weights and outputs.
