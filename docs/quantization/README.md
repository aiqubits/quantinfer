# Quantization

## Importance-weighted Q4_0

For a Llama checkpoint exported with GGML output rows, supply
`--imatrix IMATRIX.gguf` with `--gguf-layout ggml --quantize q4_0`.
Use a GGUF calibration artifact from
[llama-imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix).

The converter also accepts a JSON file with `--imatrix CHANNEL_WEIGHTS.json`.
The JSON maps each quantized projection name, such as
`blk.0.attn_q.weight`, to an array of input-channel weights. Use the mean
square of each input activation over the calibration samples. Each array
must match the projection's input width and contain finite, nonnegative
weights with at least one positive entry.

The converter evaluates several clipping points for each Q4_0 block and
keeps the scale with the smallest weighted reconstruction error. It
records the SHA-256 of the calibration input in the GGUF metadata. The
result uses the standard packed Q4_0 format.
