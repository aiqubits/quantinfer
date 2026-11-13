# Dependency boundaries

```text
applications and bindings
  -> quantinfer facade
  -> runtime / quantization / model / format
  -> core / kernel contracts
  -> hardware backends
```

Core crates must not depend on vendor SDKs, HTTP servers, Candle, or WASI-NN. Format crates must expose generic tensor archives and must not depend on model architecture modules. `quantinfer-map` is a leaf used only by the format crate. Deployment adapters must remain optional features.
