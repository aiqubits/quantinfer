# QIF conformance coverage

The wire profile and the implementation test matrix are separate. Passing the
matrix below does not prove every possible artifact or deployment correct.

| Path | Executed checks | Scope |
| --- | --- | --- |
| Rust memory/file/PACKAGE | Encoding validation, hashes, finite values, alias, shard ranges, malformed input | Eight core encodings and none/zstd/LZ4 round trips; fixed fixtures and targeted regressions |
| Rust mmap | Same validated memory reader over a read-only mapping | Unsafe constructor requires backing-file immutability; borrowed uncompressed views; opening alone does not verify payload |
| Rust/C/Python golden reports | Header bytes, parsed string/shape/tensor/chunk fields, section SHA-256, chunk/tensor digests, payload bytes | One unsigned/uncompressed F32 tensor containing `[1.0, 2.0]`, original and reordered control sections |
| Cross-language negative fixtures | Payload corruption, overflowing index length, truncation, oversized rank with consistent root | All three processes must fail; this is not a general invalid-artifact equivalence proof |
| Independent Python core profiles / native Rust | Full reports of model, typed QMB (floating IEEE bits), index records/chunks and reconstructed bytes; all eight encodings × two layouts, aliases, rank-zero scalar, three codecs × unsigned SINGLE/signed SINGLE/PACKAGE | 106 cases including relocated controls, empty artifacts, multi-chunk/multi-shard reconstruction, repaired semantic failures and pinned-key trust policy; C has not yet joined this expanded matrix |
| Linux resource limits | Six 64 MiB tensor/alias fixtures verified through both native and runtime file-backed readers in separate processes with a 48 MiB address-space limit; unknown overlapping names fail cleanly | SINGLE/PACKAGE with none/zstd/LZ4, writer-default chunks; not a universal peak-memory bound |
| Runtime loading | Control-only open, cold-corruption isolation, per-tensor raw hashes, shared alias cache, typed reads without retained raw bytes, storage budget preflight | Fixed local inference/encoding tests and explicit real-model CI profile/codec matrix; not layer-level out-of-core execution |
| Instrumented fuzz | Reader, global PACKAGE index, PACKAGE JSON/QMB/path/shard mutations; default runtime comparison; see [fuzz/README.md](../../fuzz/README.md) | Three bounded ASan targets; quick smoke plus scheduled time-budget runs, saved corpus/log/crash artifacts; not exhaustive safety proof |

The C program is independent of QuantInfer and Rust. It uses OpenSSL libcrypto
for SHA-256 and is compiled with address/undefined-behavior sanitizers in CI.
The original Python program uses only the standard library. Both original tools are golden-profile
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

The separate `qif_reference_reader.py` and `qif_reference/` package implement
the expanded Python core profile independently of QuantInfer. They do not use
the Rust runtime, its C binding, NumPy or a QuantInfer Python package. The
standard-library implementation parses canonical JSON, all frozen QMB types,
QIDX, descriptors, aliases, package paths/topology and stored/raw checksums.
Raw LZ4 decoding is independent Python code; zstd uses the system `libzstd`
streaming API with the same 128 MiB window ceiling. Ed25519 verification uses
the system OpenSSL executable, independently of Rust's ed25519-dalek.

This is a bounded conformance tool: artifact and reconstructed physical tensor
bytes are each limited to 64 MiB. It eagerly reconstructs data and reports hex;
that limit is not a process RSS bound or a production large-model loader.
Unsigned/signed SINGLE and unsigned PACKAGE are accepted. An embedded key only
reports `publisher_trust=not_evaluated`; explicit `--trusted-key HEX` pins an
externally supplied public key and requires its valid signature. It rejects
unsigned artifacts and PACKAGE under that policy. Key distribution/revocation,
PACKAGE signatures, remote transport and arbitrary filesystem races are not
tested or implemented by this tool.

On a system with OpenSSL 3 and libzstd installed:

```sh
cargo build --locked -p quantinfer-format --example qif_profile_fixtures --example qif_profile_report
python3 tests/conformance/qif_profile_cross_language.py \
  --generator "$PWD/target/debug/examples/qif_profile_fixtures" \
  --rust-reader "$PWD/target/debug/examples/qif_profile_report"
```

The 106 cases include eight fully rehashed non-finite encoding fixtures, broken
signature bytes/key/envelope and a new authenticated root with an old signature.
The PACKAGE cases cover traversal, absolute paths, symlink escape, duplicate
JSON/shards, coordinates, shard lists/ranges/sizes/digests and overlapping data.
Every valid report must agree; invalid cases require a normal failure exit,
never a crash or sanitizer error. The original three-language golden comparison
remains in CI. Expanded C support is still outstanding; these tests do not
justify claiming universal malformed-input equivalence or all acceptance done.

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
