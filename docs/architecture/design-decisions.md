# Design decisions

1. Native Rust is the primary execution target.
2. WGPU is a backend, not the definition of the core runtime.
3. WASI-NN is a deployment adapter, not a tensor library.
4. Candle is an optional interoperability adapter, not a core dependency.
5. GGUF and SafeTensors are compatibility formats; QIF remains experimental until its specification and conformance tests stabilize.
6. Cross-device numerical tolerance is tested explicitly; bitwise identity across all GPU implementations is not assumed.
