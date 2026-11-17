# QIF fuzz targets

The targets require cargo-fuzz 0.13.2 and nightly-2026-07-21 with `rust-src`.
`cargo run` alone does not provide the coverage instrumentation and ASan used
by these checks.

```sh
cargo install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-07-21 fuzz run qif_v2_reader -- -max_len=65536 -rss_limit_mb=1024 -timeout=5
cargo +nightly-2026-07-21 fuzz run qif_v2_package -- -max_len=65536 -rss_limit_mb=1024 -timeout=5
```

`qif_v2_reader` compares memory and file-backed parsing, verification and range
results, and checks the mapped reader over its exclusively owned scratch file.
The mapping is dropped before rewriting that file; concurrent modification of
a live file mapping is outside the unsafe mmap constructor's contract.
It tests both the original input and a
copy with the control root hash recomputed, so control-section mutations can
reach the parsers. Signed inputs still undergo signature validation. Payload
operations are limited to a total declared tensor length of 4 MiB; larger
inputs exercise opening only. Valid none/zstd/LZ4 seeds reach the codecs.

`qif_v2_package` mutates the global binary index in a small package with fixed
weights, model manifest and metadata. It recomputes package control hashes,
then exercises `PackageReader::open`, `read_range` and `verify`. This target
does not fuzz arbitrary package paths, model JSON or shard contents; those
boundaries also require dedicated regression tests and future corpus work.

Both targets cap input size at 1 MiB and use process-local temporary files.
CI copies the tracked seeds to temporary corpora, runs 1024 iterations per
target with a 64 KiB generation limit, and requires nonzero coverage in the
log. This is a bounded instrumented smoke check, not a claim that fuzzing is
exhaustive. Keep deterministic regressions in the format crate when a failure
is found; keep the minimized input in the named seed corpus as well.
