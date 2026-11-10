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

## Python

Python 3.9 and later can use the same shared library. Build the library with
`cargo build --release --locked -p quantinfer-c` and install the Python package
with `python -m pip install ./bindings/python`. Set `QUANTINFER_LIBRARY` to the
path of the built library, or pass the path as `Model(..., library=path)`.

```python
from quantinfer import Model

with Model("model.gguf") as model:
    print(model.generate("The capital of France is", max_new_tokens=8))
```

The package also exposes `tokenize`, `detokenize`, and `generate_tokens`.
Each generation call starts a separate decoding session. The Python package
is installed separately from the native library.
