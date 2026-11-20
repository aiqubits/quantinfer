# QIF conformance coverage

The wire profile and the implementation test matrix are separate. Passing the
matrix below does not prove every possible artifact or deployment correct.

| Path | Executed checks | Scope |
| --- | --- | --- |
| Rust memory/file/PACKAGE | Encoding validation, hashes, finite values, alias, shard ranges, malformed input | Eight core encodings and none/zstd/LZ4 round trips; fixed fixtures and targeted regressions |
| Rust mmap | Same validated memory reader over a read-only mapping | Unsafe constructor requires backing-file immutability; borrowed uncompressed views; opening alone does not verify payload |
| Rust/C/Python golden reports | Header bytes, parsed string/shape/tensor/chunk fields, section SHA-256, chunk/tensor digests, payload bytes | One unsigned/uncompressed F32 tensor containing `[1.0, 2.0]`, original and reordered control sections |
| Cross-language negative fixtures | Payload corruption, overflowing index length, truncation, oversized rank with consistent root | All three processes must fail; this is not a general invalid-artifact equivalence proof |
| Independent C/Python core profiles / native Rust | Full reports of model, typed QMB (floating IEEE bits), index records/chunks and reconstructed bytes; all eight encodings × two layouts, aliases, rank-zero scalars, three codecs × unsigned SINGLE/signed SINGLE/PACKAGE | 180 curated cases plus 12 reader corpus seeds: relocated controls, empty artifacts, multi-chunk/multi-shard reconstruction, repaired semantic failures, Unicode/NUL keys, report budgets and pinned-key trust policy; C runs with ASan/UBSan |
| Linux resource limits | Six 64 MiB tensor/alias fixtures, including a 64-shard PACKAGE, verified through both native and runtime readers in separate processes capped at 48 MiB of address space and 32 file descriptors; unknown overlapping names fail cleanly | SINGLE/PACKAGE with none/zstd/LZ4 and writer-default chunks |
| Runtime loading | Control-only open, cold-corruption isolation, per-tensor raw hashes, shared alias cache, typed reads without retained raw bytes, storage budget preflight | Fixed local inference/encoding tests and explicit real-model CI profile/codec matrix; not layer-level out-of-core execution |
| Instrumented fuzz | Reader, global PACKAGE index, PACKAGE JSON/QMB/path/shard mutations; default runtime comparison; see [fuzz/README.md](../../fuzz/README.md) | Three bounded ASan targets; quick smoke plus scheduled time-budget runs, saved corpus/log/crash artifacts; not exhaustive safety proof |

The original C program is independent of QuantInfer and Rust. It uses OpenSSL libcrypto
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

The separate C implementation in `qif_c_reference/` parses and validates the
same core profiles without QuantInfer, Rust or Python linkage. It uses JSON-C
with canonical/schema checks, system liblz4/libzstd and OpenSSL's SHA-256 and
Ed25519 APIs. It preserves embedded NULs in strings and object keys, including
the distinction between an ordinary key and a longer key containing NUL.
The [C tool notes](../../tests/conformance/qif_c_reference/README.md) describe
its internal key encoding; that is not an addition to the QIF wire format.
The C/Python/native CI matrix runs on Ubuntu 24.04 and the current Ubuntu image
and prints dependency versions. An independently validated ASCII parser buffer
avoids JSON-C 0.18's signed-char strict-mode UTF-8 bug; strict mode and canonical
comparison of the original artifact bytes remain enabled. BMP/non-BMP model
IDs, checksum keys, QMB keys and values are part of the fixtures.

These are bounded conformance tools: artifact bytes and the sum of reported
tensor raw bytes (including every alias) are each limited to 64 MiB. They
eagerly reconstruct data and report hex;
that limit is not a process RSS bound or a production large-model loader.
Unsigned/signed SINGLE and unsigned PACKAGE are accepted. An embedded key only
reports `publisher_trust=not_evaluated`; explicit `--trusted-key HEX` pins an
externally supplied public key and requires its valid signature. It rejects
unsigned artifacts and PACKAGE under that policy. Key distribution/revocation,
PACKAGE signatures, remote transport and arbitrary filesystem races are not
tested or implemented by these tools. The alias budget fixture is valid QIF,
deliberately rejected by this smaller reporting profile; it is not a new wire
limit or a requirement imposed on the production loader.

On a 64-bit POSIX system with development packages for OpenSSL 3, JSON-C,
liblz4 and libzstd installed:

```sh
cargo build --locked -p quantinfer-format --example qif_profile_fixtures --example qif_profile_report
cc -std=c11 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined \
  -fno-omit-frame-pointer tests/conformance/qif_c_reference/*.c \
  $(pkg-config --cflags --libs json-c liblz4 libzstd openssl) -o /tmp/qif-c-reference
python3 tests/conformance/qif_profile_cross_language.py \
  --generator "$PWD/target/debug/examples/qif_profile_fixtures" \
  --rust-reader "$PWD/target/debug/examples/qif_profile_report" \
  --c-reader /tmp/qif-c-reference --corpus "$PWD/fuzz/corpus/qif_v2_reader"
```

The 180 curated cases include eight fully rehashed non-finite encoding fixtures, broken
signature bytes/key/envelope and a new authenticated root with an old signature.
The PACKAGE cases cover nested Unicode paths, rejection of colons and backslashes
in existing filenames, traversal, absolute paths, symlink escape, duplicate
JSON/shards, coordinates, shard lists/ranges/sizes/digests and overlapping data.
Every valid report must agree; invalid cases require a normal failure exit,
never a crash or sanitizer error. The original three-language golden comparison
remains in CI. Existing corpus seeds additionally compare normal rejection or
full reports with the native oracle. These tests do not
justify claiming universal malformed-input equivalence or all acceptance done.

The SINGLE signature profile requires canonical public-key/R encodings,
non-small-order points and a canonical scalar. Rust uses
[`VerifyingKey::verify_strict`](https://docs.rs/ed25519-dalek/2.2.0/ed25519_dalek/struct.VerifyingKey.html#method.verify_strict)
with an additional public-key re-encoding check; C/Python apply the same
canonical/point/scalar preflight before OpenSSL verification. The matrix
rejects the identity-key/identity-R/zero-S forgery, including with that key
explicitly pinned, all eight small-order points and their encoding variants,
noncanonical points/scalars and altered roots. These checks do not let an
attacker impersonate a properly pinned normal signing key; they prevent weak
keys from being reported as verified. Normal generated-key signatures still
agree across all three codecs.

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

Runtime layout tests are separate from physical-byte conformance reports.
Eight encodings use a non-square 256×32 matrix, both layouts and both dimension
profiles, with independent F64 matrix arithmetic and unchanged quantized blocks.
Complete non-symmetric Qwen3/Llama fixtures additionally exercise 72 combinations
of dimension profile, uniform/mixed layout, codec and owned/file/PACKAGE source,
each through the default factory and direct backend entry point. They cover
prefill, decode, KV lengths and raw-cache release. The GGML dimension profile
reverses logical axes in the packed view; changing its layout label alone need
not change physical bytes. Coordinate rules are in the [format plan](qif-v2-final-plan.md#521-矩阵坐标与兼容视图).

The dimension-order field is structural metadata. The native writer and all
three independent reader implementations reject a non-boolean top-level
`quantinfer.qif.ggml_dimension_order` during control validation. Four rehashed
negative controls exercise integer/string/map/array types, and two positive
controls retain false/true. Nested keys under an unknown hint remain ordinary
metadata. The current matrix is 180 curated cases plus 12 corpus seeds, or
192 checks per CI environment.

Larger physical alignment is verified independently for none/zstd/LZ4, with
4 KiB, 64 KiB and 2 MiB SINGLE sections/chunks and the same index/payload
in PACKAGE shards. Eighteen positive and eighteen rehashed unaligned,
overlapping or backward-offset cases extend the curated matrix. Tensor
logical values and raw digests stay unchanged; padding is outside chunk
lengths and content hashes. Readers locate explicit offsets rather than
requiring the next chunk at the smallest possible 4 KiB boundary. Native
Rust tests additionally compare memory/file/mmap/default loading, aliases
and ranges spanning both chunks. Writers retain their existing 4 KiB default.
