# Quantization

## Collect calibration weights

Run the model on representative prompts to measure each input channel's
mean squared activation. Put one JSON string or one object with a
`text` string on each line of the dataset.

```sh
quantinfer collect-imatrix --input model-f32.gguf --dataset prompts.jsonl \
  --output imatrix.json --max-sequence-length 512
```

Collection supports Qwen3 and Llama, with an embedded tokenizer and
GGUF, QIF or a verified package as input. It starts a fresh KV cache
for each prompt and records attention and feed-forward projection inputs.
Each token contributes one observation. The default context limit is
512 tokens or the model limit if it is smaller. The collector rejects
oversized prompts and writes the output only after all samples succeed.

## Importance-weighted Q4_0

For a Llama or Qwen3 checkpoint exported with GGML output rows, supply
`--imatrix IMATRIX.gguf` with `--gguf-layout ggml --quantize q4_0`.
Use a GGUF calibration artifact from
[llama-imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix).

Pass the collector's output to the converter with `--imatrix imatrix.json`.
The JSON maps each quantized projection name, such as
`blk.0.attn_q.weight`, to an array of input-channel weights. Use the mean
square of each input activation over the calibration samples. Each array
must match the projection's input width and contain finite, nonnegative
weights with at least one positive entry.

For Qwen3, use `--tokenizer-pre qwen2` with the checkpoint's tokenizer.

```sh
quantinfer-convert --config config.json --weights model.safetensors \
  --tokenizer tokenizer.json --tokenizer-pre qwen2 --gguf-layout ggml \
  --quantize q4_0 --imatrix imatrix.json --output model-q4_0.gguf
```

The converter evaluates several clipping points for each Q4_0 block and
keeps the scale with the smallest weighted reconstruction error. It
records the SHA-256 of the calibration input in the GGUF metadata. The
result uses the standard packed Q4_0 format.
