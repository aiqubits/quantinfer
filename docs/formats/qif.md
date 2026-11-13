# QIF version 1

QIF is an optional QuantInfer container for a model manifest, typed metadata,
and tensor payloads. The format crate exposes a streaming writer and a reader
that verifies the full artifact before returning borrowed tensor bytes.

The converter imports a supported GGUF artifact and preserves its typed
metadata and quantized weights. CPU generation accepts QIF files directly.
Opening a QIF or GGUF path maps the file read-only and checks it there.
Decoding a tensor still copies values into a runtime tensor.

```sh
quantinfer-convert from-gguf --input model.gguf --output model.qif
quantinfer generate --input model.qif --prompt "The capital of France is" --max-new-tokens 8
```

## Container

All integers use little-endian encoding. Offsets are measured from the start
of the file.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | Magic bytes `51 49 46 00` |
| 4 | 4 | Unsigned container version, currently 1 |
| 8 | 8 | Unsigned UTF-8 manifest byte length |
| 16 | 8 | Unsigned metadata byte length |
| 24 | 32 | Raw SHA-256 digest of the manifest and metadata concatenated |
| 56 | Variable | ModelManifest JSON, then metadata |

The manifest uses its own schema version. The combined header sections are
limited to 64 MiB. Tensor data follows the header in manifest order. Each
payload starts on a 64-byte boundary and all padding bytes are zero. The
final payload ends at EOF. For an empty tensor list, the metadata ends at EOF.
The writer records the lowercase hexadecimal SHA-256 of every tensor in
`tensors[].checksum_sha256`. The reader requires these checksums and rejects
unknown container versions, unsupported descriptors, corrupt padding,
truncated tensors, and trailing data.

## Metadata

The metadata section starts with a 64-bit entry count. Each entry contains a
nonempty, unique key string, a 32-bit type tag, and a typed value. Strings
contain a 64-bit byte length followed by UTF-8 bytes.

| Tag | Value |
| --- | --- |
| 4 | Unsigned 32-bit integer |
| 5 | Signed 32-bit integer |
| 6 | IEEE-754 binary32 |
| 7 | One-byte boolean with zero for false |
| 8 | String |
| 9 | Array with a 32-bit element tag, a 64-bit count, and untagged element values |
| 10 | Unsigned 64-bit integer |
| 11 | Signed 64-bit integer |

Arrays have one scalar element type and contain at most 1,048,576 entries.
Metadata keys are sorted by the writer.

The boolean key `quantinfer.qif.ggml_dimension_order` records that a Llama or
Qwen3 source used GGML output rows. QIF still stores row-major descriptor
dimensions. The packed runtime adapter restores the dimension convention
expected by the GGML weight loader, which also handles Llama's RoPE channel
ordering.

## Tensor storage

Version 1 uses the canonical `TensorStorageDescriptor` fields and row-major
shape order. F32, F16 and BF16 payloads use little-endian scalar bytes.
Quantized tensors use the standard packed GGML block encodings. Blocks
follow tensor element order.

| Encoding | Elements per block | Bytes per block |
| --- | --- | --- |
| Q8_0 | 32 | 34 |
| Q4_0 | 32 | 18 |
| Q4_K | 256 | 144 |
| Q5_K | 256 | 176 |
| Q6_K | 256 | 210 |

Q4_0 stores a binary16 scale and 16 packed bytes. The low nibbles store
elements 0–15 and the high nibbles store elements 16–31. The descriptor
declares `nibble_split_half` packing.
