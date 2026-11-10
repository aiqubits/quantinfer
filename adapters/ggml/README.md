# GGML Q8_0 adapter

The optional provider calls upstream GGML to quantize F32 tensors and returns
QuantInfer Q8_0 blocks for CPU execution. The converter records the provider in
GGUF metadata. The adapter shares the native Q8_0 decoder for error reporting.

Build `ggml-base` from llama.cpp revision
`0ec4027a2a0183de87c007069adddc06e665f7de`. The GGML Conformance workflow
contains the CMake configuration and tests the external provider on Linux.
Point the linker and loader at the directory containing the built library.

```bash
export QUANTINFER_GGML_LIB_DIR=/path/to/ggml/build/bin
export LD_LIBRARY_PATH="$QUANTINFER_GGML_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo run --release --locked -p quantinfer-convert --features ggml -- \
  --config config.json --weights model.safetensors --tokenizer tokenizer.json \
  --output model.gguf --quantize q8_0 --ggml
```
