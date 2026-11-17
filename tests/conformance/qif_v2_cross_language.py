#!/usr/bin/env python3
"""Compare independent wire reports and require rejection of damaged fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def commands(path, args):
    return [
        [args.rust_reader, str(path)],
        [args.c_reader, str(path), '--json'],
        [sys.executable, str(Path(__file__).with_name('qif_v2_reader.py')), '--input', str(path), '--json'],
    ]


def refresh_root(data):
    digest = hashlib.sha256()
    for offset in (24, 40, 56):
        start, size = struct.unpack_from('<QQ', data, offset)
        digest.update(data[start:start+size])
    data[88:120] = digest.digest()


def main():
    parser = argparse.ArgumentParser()
    for option in ('input', 'rust-reader', 'c-reader'):
        parser.add_argument('--'+option, required=True)
    args = parser.parse_args()
    original = Path(args.input).read_bytes()
    with tempfile.TemporaryDirectory(prefix='qif-conformance-') as directory:
        root = Path(directory)
        reordered = bytearray(original)
        manifest, metadata = [struct.unpack_from('<QQ', original, off) for off in (24, 40)]
        if max(manifest[1], metadata[1]) > 4096:
            raise ValueError('golden control sections must fit a page')
        reordered[metadata[0]:metadata[0]+manifest[1]] = original[manifest[0]:sum(manifest)]
        reordered[manifest[0]:manifest[0]+metadata[1]] = original[metadata[0]:sum(metadata)]
        struct.pack_into('<QQ', reordered, 24, metadata[0], manifest[1])
        struct.pack_into('<QQ', reordered, 40, manifest[0], metadata[1])
        for name, data in [('original', original), ('reordered', reordered)]:
            path = root/(name+'.qif')
            path.write_bytes(data)
            reports = []
            for command in commands(path, args):
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
                reports.append(json.loads(result.stdout))
            if not reports[0] == reports[1] == reports[2]:
                raise AssertionError(f'{name}: Rust/C/Python wire reports differ')
        damaged = {}
        data = bytearray(original)
        payload_offset = struct.unpack_from('<Q', data, 72)[0]
        data[payload_offset+4] ^= 1
        damaged['second-float-corrupt'] = data
        data = bytearray(original)
        struct.pack_into('<Q', data, 64, 2**64-1)
        damaged['index-length-overflow'] = data
        damaged['truncated'] = original[:-1]
        data = bytearray(original)
        index = struct.unpack_from('<Q', data, 56)[0]
        shape = index+48+struct.unpack_from('<Q', data, index+16)[0]
        struct.pack_into('<I', data, shape, 33)
        refresh_root(data)
        damaged['oversized-rank-with-valid-root'] = data
        for name, data in damaged.items():
            path = root/(name+'.qif')
            path.write_bytes(data)
            for command in commands(path, args):
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    raise AssertionError(f'{name}: {command[0]} accepted damaged fixture')
    print('QIF cross-language: 2 valid layouts agree; 4 damaged fixtures rejected by all readers')


if __name__ == '__main__':
    main()
