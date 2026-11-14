# Changelog

## Unreleased

- Establish the initial QuantInfer workspace and module boundaries.
- Harden artifact parsing against non-finite values, overflowing shapes,
  duplicate metadata, unsafe package paths, and ambiguous SafeTensors JSON.
- `Shape::num_elements` now returns a checked `Result<usize>`; downstream
  callers must handle invalid overflowing shapes explicitly.
