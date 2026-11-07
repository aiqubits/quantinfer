# Benchmarks

Operator, runtime, quantization, and end-to-end performance measurements.

The first reproducible smoke benchmark covers Q8_0 quantization and CPU scalar
matrix multiplication.

```bash
cargo run -p quantinfer-xtask -- bench
```

It prints the iteration count, matrix shape, elapsed nanoseconds, and a numeric
checksum. The command is also executed by the GitHub Actions workflow.
