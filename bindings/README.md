# Language bindings

C, Python, and other language-facing APIs.

## C ABI

Build the shared library with `cargo build --release --locked -p quantinfer-c`.
Include `c/include/quantinfer.h` and link against `quantinfer_c`. A model handle
loads a GGUF file or a verified package and can generate tokens, tokenize text,
and decode token IDs. The caller owns the output buffers. The header defines
the buffer size protocol and the lifetime of diagnostic strings.

Model Conformance compiles a C caller and compares its prompt, generated
tokens, decoded text and EOS behavior against the SmolLM2 fixture.
