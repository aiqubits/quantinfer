#!/usr/bin/env python3
"""Exercise default QIF verification under a Linux address-space budget."""
import argparse
import hashlib
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile


def u64(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]


def set_limits():
    resource.setrlimit(resource.RLIMIT_AS, (48 * 1024 * 1024, 48 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def unknown_names_fixture(seed):
    """Small index with many unique, overlapping, oversized unknown names."""
    index_start, index_length = u64(seed, 56), u64(seed, 64)
    old = seed[index_start:index_start+index_length]
    count = struct.unpack_from('<I', old, 8)[0]
    string_length, shape_length, record_length = [u64(old, n) for n in (16, 24, 32)]
    strings = bytearray(1024 * 1024 + count * 8)
    for index in range(count):
        struct.pack_into('<I', strings, index*8, 1024*1024)
        strings[index*8+4:index*8+8] = f'{index:04d}'.encode('ascii')
    shapes = old[48+string_length:48+string_length+shape_length]
    record_start = 48+string_length+shape_length
    records = bytearray(old[record_start:record_start+record_length])
    for index in range(count):
        struct.pack_into('<I', records, index*120+8, index*8)
    header = bytearray(old[:48])
    struct.pack_into('<Q', header, 16, len(strings))
    new_index = header + strings + shapes + records + old[record_start+record_length:]
    payload_offset = (index_start+len(new_index)+4095)//4096*4096
    old_payload, payload_size = u64(seed, 72), u64(seed, 80)
    data = bytearray(payload_offset+payload_size)
    data[:index_start] = seed[:index_start]
    data[index_start:index_start+len(new_index)] = new_index
    data[payload_offset:] = seed[old_payload:old_payload+payload_size]
    struct.pack_into('<Q', data, 64, len(new_index))
    struct.pack_into('<Q', data, 72, payload_offset)
    digest = hashlib.sha256()
    for offset in (24, 40, 56):
        start, size = u64(data, offset), u64(data, offset+8)
        digest.update(data[start:start+size])
    data[88:120] = digest.digest()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True)
    parser.add_argument('--verifier', required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('the address-space check requires Linux RLIMIT_AS')
    with tempfile.TemporaryDirectory(prefix='qif-resource-check-') as temporary:
        root = Path(temporary)/'fixtures'
        subprocess.run([args.generator, str(root)], check=True, timeout=180)
        for name in ('none.qif', 'zstd.qif', 'lz4.qif', 'package-none', 'package-zstd', 'package-lz4'):
            for mode in ([], ['--packed']):
                result = subprocess.run([args.verifier, *mode, str(root/name)], preexec_fn=set_limits,
                                        capture_output=True, text=True, timeout=45)
                if result.returncode != 0:
                    raise AssertionError(f'{name} {mode}: verification failed with 48 MiB address space: {result.stderr}')
        malformed = root/'unknown-names.qif'
        malformed.write_bytes(unknown_names_fixture((root/'name-seed.qif').read_bytes()))
        result = subprocess.run([args.verifier, str(malformed)], preexec_fn=set_limits,
                                capture_output=True, text=True, timeout=15)
        if result.returncode != 1 or 'not in manifest' not in result.stderr:
            raise AssertionError(f'unknown names must fail cleanly before allocating all names: {result.stderr[:500]}')
    print('QIF resources: six 64 MiB tensor variants verified within 48 MiB through native and runtime readers; unknown names rejected cleanly')


if __name__ == '__main__':
    main()
