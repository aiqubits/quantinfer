#!/usr/bin/env python3
"""Independent unsigned, uncompressed F32 golden-profile wire reader."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct


def require(condition: bool) -> None:
    if not condition:
        raise ValueError('invalid or unsupported QIF golden profile')


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from('<I', data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return struct.unpack_from('<Q', data, offset)[0]


def region(data: bytes, start: int, length: int) -> bytes:
    require(start <= len(data) and length <= len(data) - start)
    return data[start:start + length]


def report(path: str) -> dict:
    with open(path, 'rb') as source:
        data = source.read(1024 * 1024 + 1)
    require(144 <= len(data) <= 1024 * 1024)
    require(data[:4] == b'QIF\0')
    require([u32(data, n) for n in (4, 8, 12, 16, 20)] == [2, 0, 144, 1, 1])
    require(all(u64(data, n) == 0 for n in (120, 128, 136)))
    bounds = [(u64(data, n), u64(data, n + 8)) for n in (24, 40, 56, 72)]
    for i, (start, length) in enumerate(bounds):
        require(start >= 144 and start % 4096 == 0)
        for other, size in bounds[:i]:
            require(start >= other + size or other >= start + length)
    manifest, metadata, index, payload_section = [region(data, *entry) for entry in bounds]
    require(sum(bounds[3]) == len(data))
    require(hashlib.sha256(manifest + metadata + index).digest() == data[88:120])
    model = json.loads(manifest)
    require(model['model_id'] == 'qif-v2-fixture')
    require(index[:4] == b'QIDX' and u32(index, 4) == 1)
    require([u32(index, 8), u32(index, 12)] == [1, 1])
    lengths = [u64(index, offset) for offset in (16, 24, 32, 40)]
    strings_len, shapes_len, records_len, chunks_len = lengths
    strings = region(index, 48, strings_len)
    shapes = region(index, 48 + strings_len, shapes_len)
    require(records_len == 120 and chunks_len == 72)
    record_start = 48 + strings_len + shapes_len
    record = region(index, record_start, records_len)
    chunk = region(index, record_start + records_len, chunks_len)
    require(record_start + records_len + chunks_len == len(index))
    refs = [u32(record, 8), u32(record, 12)]
    name_length = u32(region(strings, refs[0], 4), 0)
    name = region(strings, refs[0] + 4, name_length).decode('utf-8')
    shape = region(shapes, refs[1], 12)
    require(name == 'weight' and u32(shape, 0) == 1 and u64(shape, 4) == 2)
    encoding = [u32(record, n) for n in (16, 20, 24, 28, 32)]
    flags = [u32(record, n) for n in (36, 40, 44, 112, 116)]
    tensor_lengths = [u64(record, n) for n in (48, 56, 64)]
    chunk_values = [u32(chunk, 0), u32(chunk, 4)] + [u64(chunk, n) for n in (8, 16, 24, 32)]
    require(u64(record, 0) == 0 and encoding == [1, 1, 0, 1, 0])
    require(flags == [0, 0, 1, 0, 0] and tensor_lengths == [8, 8, 0])
    require(u64(record, 104) == 2**64 - 1 and chunk_values == [0, 0, 0, 8, 0, 8])
    payload = region(payload_section, chunk_values[4], chunk_values[5])
    digest = hashlib.sha256(payload).digest()
    require(digest == chunk[40:72] == record[72:104])
    require(digest.hex() == model['tensors'][0]['checksum_sha256'])
    require(payload == struct.pack('<ff', 1, 2))
    return {
        'header_hex': data[:144].hex(),
        'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
        'metadata_sha256': hashlib.sha256(metadata).hexdigest(),
        'index_sha256': hashlib.sha256(index).hexdigest(),
        'tensor_sha256': record[72:104].hex(), 'chunk_sha256': chunk[40:72].hex(),
        'index_counts': [1, 1], 'table_lengths': lengths,
        'tensor_id': u64(record, 0), 'name': name, 'shape': [u64(shape, 4)],
        'refs': refs, 'encoding': encoding, 'flags': flags, 'lengths': tensor_lengths,
        'alias_target': u64(record, 104), 'chunk': chunk_values, 'payload_hex': payload.hex(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = report(args.input)
    print(json.dumps(result, sort_keys=True) if args.json else 'qif-python-golden: ok')


if __name__ == '__main__':
    main()
