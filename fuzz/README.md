# QIF fuzz targets

The targets require cargo-fuzz 0.13.2 and nightly-2026-07-21 with `rust-src`.
`cargo run` alone does not provide the coverage instrumentation and ASan used
by these checks.

```sh
cargo install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-07-21 fuzz run qif_v2_reader -- -max_len=65536 -rss_limit_mb=1024 -timeout=5
cargo +nightly-2026-07-21 fuzz run qif_v2_package -- -max_len=65536 -rss_limit_mb=1024 -timeout=5
cargo +nightly-2026-07-21 fuzz run qif_v2_package_controls -- -max_len=65536 -rss_limit_mb=1024 -timeout=5
```

`qif_v2_reader` compares memory and file-backed parsing, verification and range
results, checks the default runtime's file-backed accessor and raw hash validation,
and checks the mapped reader over its exclusively owned scratch file.
The mapping is dropped before rewriting that file; concurrent modification of
a live file mapping is outside the unsafe mmap constructor's contract.
It tests both the original input and a
copy with the control root hash recomputed, so control-section mutations can
reach the parsers. Signed inputs still undergo signature validation. Payload
operations are limited to a total declared tensor length of 4 MiB; larger
inputs exercise opening only. Valid none/zstd/LZ4 seeds reach the codecs.

`qif_v2_package` mutates the global binary index in a small package with fixed
weights, model manifest and metadata. It recomputes package control hashes,
then exercises `PackageReader::open`, `read_range` and `verify`.

`qif_v2_package_controls` starts with three small packages (none/zstd/LZ4), each
with a physical F32 tensor and an alias. Its first byte selects the codec;
the second byte selects raw package JSON, typed model/package JSON with repaired
control hashes, QMB metadata, shard bytes, or paths. Path mode starts with a byte
selecting metadata/index/shard path, followed by UTF-8 path bytes. It writes only
fixed scratch filenames, and tests root escapes, missing files and an outside-root
symlink without using fuzzed paths for writes. The target compares native and
default runtime verification and bounds payload operations to 4 MiB. It does not
simulate concurrent filesystem replacement, all shard topologies, or publisher
trust. Its canonical-JSON mirror is test infrastructure for hash repair, not a
second production reader. Regenerate its baseline seeds with:

```sh
cargo +nightly-2026-07-21 run --locked --manifest-path fuzz/Cargo.toml \
  --example package_controls_corpus -- fuzz/corpus/qif_v2_package_controls
```

All targets cap input size at 1 MiB and use process-local temporary files.
CI copies the tracked seeds to temporary corpora, runs 1024 iterations per
target with a 64 KiB generation limit, and requires nonzero coverage in the
log. This is a bounded instrumented smoke check, not a claim that fuzzing is
exhaustive. Keep deterministic regressions in the format crate when a failure
is found; keep the minimized input in the named seed corpus as well.

[QIF Sustained Fuzz](../.github/workflows/qif-fuzz.yml) runs three separate ASan
workers nightly for 900 seconds per target, and 60 seconds on relevant pushes.
Manual runs accept 60–3600 seconds. Corpora are restored across runs with a
target-specific cache; each run uploads logs, corpus and any crash artifacts
(30-day retention). Smoke CI also uploads failure artifacts (14 days). A failing
run remains a failure; regressions and minimized crash seeds must be committed
after triage. Time budgets and nonzero coverage establish bounded execution,
not exhaustive safety or correctness.
