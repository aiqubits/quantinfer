# QIF conformance coverage

The wire profile and the implementation test matrix are separate. Passing the
matrix below does not prove every possible artifact or deployment correct.

| Path | Executed checks | Scope |
| --- | --- | --- |
| Rust memory/file/PACKAGE | Encoding validation, hashes, finite values, alias, shard ranges, malformed input | Eight core encodings and none/zstd/LZ4 round trips; fixed fixtures and targeted regressions |
| Rust mmap | Same validated memory reader over a read-only mapping | Uncompressed borrowed views; opening a mapping alone does not verify payload |
| Rust/C/Python golden reports | Header bytes, parsed string/shape/tensor/chunk fields, section SHA-256, chunk/tensor digests, payload bytes | One unsigned/uncompressed F32 tensor containing `[1.0, 2.0]`, original and reordered control sections |
| Cross-language negative fixtures | Payload corruption, overflowing index length, truncation, oversized rank with consistent root | All three processes must fail; this is not a general invalid-artifact equivalence proof |
| Linux resource limits | Six 64 MiB tensor/alias fixtures verified in separate processes with a 48 MiB address-space limit; overlapping unknown index names fail cleanly | SINGLE/PACKAGE with none/zstd/LZ4, writer-default chunk sizes; not a universal peak-memory bound |
| Instrumented fuzz | See [fuzz/README.md](../../fuzz/README.md) | Input/output/time/RSS bounds apply; CI smoke is not long-running fuzzing |

The C program is independent of QuantInfer and Rust. It uses OpenSSL libcrypto
for SHA-256 and is compiled with address/undefined-behavior sanitizers in CI.
The Python program uses only the standard library. Both are golden-profile
wire tools, not general JSON/QMB semantic validators, compressed readers,
PACKAGE readers or signed-artifact trust implementations. The production C
binding is a separate optional interface to the native Rust runtime.

On a system with a C compiler and OpenSSL development headers:

```sh
cargo run --locked -p quantinfer-format --example qif_v2_fixture -- /tmp/golden.qif
cargo build --locked -p quantinfer-format --example qif_v2_report
cc -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined \
  -fno-omit-frame-pointer tests/conformance/qif_v2_reader.c -lcrypto -o /tmp/qif-c-reader
python3 tests/conformance/qif_v2_cross_language.py --input /tmp/golden.qif \
  --rust-reader target/debug/examples/qif_v2_report --c-reader /tmp/qif-c-reader
```

The fixture generator also accepts `zstd` or `lz4` after the output path for
Rust codec/fuzz seeds. Those variants are deliberately outside the C/Python
golden profile. Add fixtures and independent parsing before extending the
claimed cross-language coverage.

The Linux resource check is independent of the cross-language golden profile:

```sh
cargo build --locked -p quantinfer-format --example qif_verify --example qif_resource_fixture
python3 tests/conformance/qif_resource_bounds.py \
  --generator "$PWD/target/debug/examples/qif_resource_fixture" \
  --verifier "$PWD/target/debug/examples/qif_verify"
```

Compressed chunks are bounded by the current reader resource profile (128 MiB
decoded, 129 MiB stored; zstd window at most 128 MiB). The writer uses at most
4 MiB of raw data per compressed chunk. The 48 MiB test uses those writer
defaults; it does not assert that a maximum-size allowed chunk fits that budget.
