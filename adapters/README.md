# External adapters

Adapters connect QuantInfer to external Rust inference ecosystems without making them core dependencies.

## Candle

Enable the `candle` feature on `quantinfer` to use `quantinfer::candle`.
`to_candle` and `from_candle` copy floating tensors while preserving shape
and element order. Imports from Candle use F32 storage in QuantInfer.

`packed_from_qif` loads validated Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K payloads
into Candle without dequantizing them. Packed blocks must fit within the
last dimension. The resulting Candle tensor retains the stored row order.
