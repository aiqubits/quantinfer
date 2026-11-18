# Independent QIF C conformance reader

This standalone C11 CLI links JSON-C, liblz4, libzstd and OpenSSL. It never calls
QuantInfer, Rust or a Python binding. Build and matrix commands are in
[QIF conformance](../../../docs/formats/qif-conformance.md).

The executable accepts a SINGLE file or PACKAGE directory and writes a JSON
wire report. It validates canonical/schema JSON, QMB, the eight frozen storage
encodings and two layouts, aliases, chunk ownership/order/ranges, stored/raw
SHA-256, finite scalars/scales, package paths/topology/shard hashes and SINGLE
Ed25519 envelopes (canonical/non-small-order A/R and canonical S, then the
OpenSSL verification equation). `--trusted-key HEX` additionally requires a valid signature
from an externally pinned key. An embedded key alone establishes no trust.

The test profile requires a 64-bit POSIX host. Artifact bytes and the sum of
reported raw tensor bytes, counting aliases, are each limited to 64 MiB.
Decoded hex reports, library allocations and JSON objects still consume extra
memory; this is not an RSS cap or a streaming production loader. Files must
remain unchanged throughout a run. PACKAGE signatures and key distribution or
revocation are outside the frozen profile and this executable.

JSON-C stores object keys as C strings. The implementation encodes a key
internally if it contains NUL or starts with the reserved internal prefix.
Encoding prefix-bearing keys too makes the transformation injective. Canonical
serialization restores the original UTF-8 key bytes. This transformation only
exists inside this test tool: the artifact and output report retain the original
keys. Tests contain NUL keys, nested maps, ordinary prefix-sharing keys,
prefix/hex-like keys and duplicate keys. String values retain explicit lengths.

The resource list releases all owned buffers, JSON references, files, crypto and
decoder contexts on both successful output and ordinary parse/validation
failure. CI compiles with ASan/UBSan and treats sanitizer reports, crashes or
tracebacks as failures, including on deliberately invalid inputs. Sanitizers
instrument this C code, not the distribution's prebuilt shared libraries.
