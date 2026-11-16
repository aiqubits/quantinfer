#!/usr/bin/env python3
"""Dependency-free QIF2-SINGLE golden reader for the uncompressed core profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    data = open(args.input, "rb").read()
    assert data[:4] == b"QIF\0"
    assert u32(data, 4) == 2
    assert u32(data, 8) == 0
    assert u32(data, 12) == 144
    assert u32(data, 16) == 1
    sections = [(u64(data, 24), u64(data, 32)),
                (u64(data, 40), u64(data, 48)),
                (u64(data, 56), u64(data, 64)),
                (u64(data, 72), u64(data, 80))]
    assert u64(data, 120) == 0 and u64(data, 128) == 0
    manifest = data[sections[0][0]:sections[0][0] + sections[0][1]]
    metadata = data[sections[1][0]:sections[1][0] + sections[1][1]]
    index = data[sections[2][0]:sections[2][0] + sections[2][1]]
    assert hashlib.sha256(manifest + metadata + index).digest() == data[88:120]
    model = json.loads(manifest)
    assert model["model_id"] == "qif-v2-fixture"
    assert index[:4] == b"QIDX" and u32(index, 4) == 1
    assert u32(index, 8) == 1 and u32(index, 12) == 1
    strings_len, shapes_len = u64(index, 16), u64(index, 24)
    records_len, chunks_len = u64(index, 32), u64(index, 40)
    assert records_len == 120 and chunks_len == 72
    record_start = 48 + strings_len + shapes_len
    chunk_start = record_start + records_len
    record = index[record_start:record_start + records_len]
    chunk = index[chunk_start:chunk_start + chunks_len]
    assert u32(record, 32) == 0 and u32(chunk, 0) == 0 and u32(chunk, 4) == 0
    stored_offset, stored_length = u64(chunk, 24), u64(chunk, 32)
    payload_offset = sections[3][0]
    payload = data[payload_offset + stored_offset:payload_offset + stored_offset + stored_length]
    assert hashlib.sha256(payload).digest() == chunk[40:72]
    assert hashlib.sha256(payload).hexdigest() == model["tensors"][0]["checksum_sha256"]
    print("qif2-python-golden: ok")


if __name__ == "__main__":
    main()
