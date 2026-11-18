#!/usr/bin/env python3
"""Compare independent core-profile reports and reject repaired malicious files."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

from qif_reference.schema import canonical


def u64(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]


def section(data, offset):
    start, length = struct.unpack_from('<QQ', data, offset)
    return bytes(data[start:start + length])


def repair_single(data):
    data[88:120] = hashlib.sha256(b''.join(section(data, offset) for offset in (24, 40, 56))).digest()


def locations(data):
    index = u64(data, 56)
    records = index + 48 + u64(data, index + 16) + u64(data, index + 24)
    chunks = records + u64(data, index + 32)
    return index, records, chunks


def single_cases(original):
    cases = {}

    def edit(name, offset, fmt, value, repair=True):
        data = bytearray(original)
        struct.pack_into('<' + fmt, data, offset, value)
        if repair:
            repair_single(data)
        cases[name] = data

    index, records, chunks = locations(original)
    for name, offset, fmt, value in [
        ('version', 4, 'I', 1), ('unknown-flags', 8, 'I', 1), ('hash-algorithm', 16, 'I', 9),
        ('header-reserved', 136, 'Q', 1), ('signature-offset-without-length', 120, 'Q', 4096),
        ('section-overlap', 40, 'Q', u64(original, 24)), ('unaligned-section', 40, 'Q', 4097),
        ('overflow-index-size', 64, 'Q', (1 << 64) - 1),
    ]:
        edit(name, offset, fmt, value, repair=False)
    for name, offset, fmt, value in [
        ('index-count', index + 8, 'I', 0), ('record-length', index + 32, 'Q', 0),
        ('string-ref', records + 8, 'I', (1 << 32) - 1),
        ('shape-ref', records + 12, 'I', (1 << 32) - 1),
        ('rank', index + 48 + u64(original, index + 16), 'I', 33),
        ('zero-dimension', index + 48 + u64(original, index + 16) + 4, 'Q', 0),
        ('tensor-id', records, 'Q', 1), ('encoding', records + 20, 'I', 2),
        ('scheme', records + 24, 'I', 1), ('layout', records + 28, 'I', 2),
        ('tensor-flags', records + 36, 'I', 1), ('reserved-record', records + 116, 'I', 1),
        ('raw-size', records + 48, 'Q', 1), ('stored-size', records + 56, 'Q', 1),
        ('data-offset', records + 64, 'Q', 1), ('forward-alias', records + 104, 'Q', 1),
        ('alias-has-chunks', records + 120 + 44, 'I', 1),
        ('alias-metadata', records + 120 + 32, 'I', 1),
        ('unreferenced-chunk', records + 44, 'I', 0),
        ('unknown-codec', chunks + 4, 'I', 7), ('single-shard', chunks, 'I', 1),
        ('raw-offset', chunks + 8, 'Q', 1), ('zero-chunk', chunks + 16, 'Q', 0),
        ('unaligned-chunk', chunks + 24, 'Q', 1),
        ('chunk-overflow', chunks + 24, 'Q', (1 << 64) - 4095),
    ]:
        edit(name, offset, fmt, value)
    data = bytearray(original)
    data[records + 72] ^= 1
    repair_single(data)
    cases['tensor-checksum-versus-manifest'] = data
    data = bytearray(original)
    data[chunks + 40] ^= 1
    repair_single(data)
    cases['stored-checksum'] = data
    data = bytearray(original)
    data[u64(data, 72) + 4] ^= 1
    cases['payload-corruption'] = data
    cases['truncated'] = original[:-1]
    cases['trailing'] = original + b'\0'

    # Metadata syntax, ordering and finite values with a consistent control root.
    metadata_start = u64(original, 40)
    raw_metadata = section(original, 40)
    for name, before, after in [
        ('metadata-nan', struct.pack('<If', 6, 0.1), struct.pack('<II', 6, 0x7fc00000)),
        ('metadata-bool', struct.pack('<IB', 7, 1), struct.pack('<IB', 7, 2)),
        ('metadata-order', b'a.u32', b'z.u32'),
        ('metadata-duplicate', b'b.i32', b'a.u32'),
        ('metadata-utf8', b'h.string', b'h.strin\xff'),
    ]:
        offset = metadata_start + raw_metadata.index(before)
        data = bytearray(original)
        data[offset:offset + len(before)] = after
        repair_single(data)
        cases[name] = data
    edit('metadata-root-tag', metadata_start, 'I', 9)
    edit('metadata-count', metadata_start + 4, 'Q', (1 << 64) - 1)

    # Every physical encoding gets a non-finite scalar/scale, with ALL applicable
    # hashes repaired. This must fail value validation, not an earlier hash check.
    manifest_start = u64(original, 24)
    model = json.loads(section(original, 24))
    tensor_count = struct.unpack_from('<I', original, index + 8)[0]
    tested = set()
    for n in range(tensor_count):
        record = records + n * 120
        dtype = struct.unpack_from('<I', original, record + 16)[0]
        alias = u64(original, record + 104)
        if alias != (1 << 64) - 1 or dtype in tested:
            continue
        tested.add(dtype)
        data = bytearray(original)
        chunk_id = struct.unpack_from('<I', data, record + 40)[0]
        chunk = chunks + chunk_id * 72
        start = u64(data, 72) + u64(data, chunk + 24)
        scale_offset = 208 if dtype == 7 else 0
        fmt, bits = ('I', 0x7f800000) if dtype == 1 else ('H', 0x7f80 if dtype == 3 else 0x7c00)
        struct.pack_into('<' + fmt, data, start + scale_offset, bits)
        digest = hashlib.sha256(data[start:start + u64(data, chunk + 32)]).digest()
        data[chunk + 40:chunk + 72] = digest
        modified_model = copy.deepcopy(model)
        # The uncompressed SINGLE writer uses one chunk per physical tensor.
        for m in range(tensor_count):
            other = records + m * 120
            if m == n or u64(data, other + 104) == n:
                data[other + 72:other + 104] = digest
                modified_model['tensors'][m]['checksum_sha256'] = digest.hex()
        replacement = canonical(modified_model)
        assert len(replacement) == u64(data, 32)
        data[manifest_start:manifest_start + len(replacement)] = replacement
        repair_single(data)
        cases[f'nonfinite-encoding-{dtype}-all-hashes-repaired'] = data
    return cases


def reordered(original):
    """Relocate whole controls/signature, preserving their authenticated bytes."""
    output = bytearray(original[:144])

    def append(data):
        offset = (len(output) + 4095) // 4096 * 4096
        output.extend(bytes(offset - len(output)))
        output.extend(data)
        return offset

    for field in (56, 24, 40):
        data = section(original, field)
        offset = append(data)
        struct.pack_into('<QQ', output, field, offset, len(data))
    if u64(original, 128):
        data = section(original, 120)
        offset = append(data)
        struct.pack_into('<QQ', output, 120, offset, len(data))
    payload = section(original, 72)
    offset = append(payload)
    struct.pack_into('<QQ', output, 72, offset, len(payload))
    return output


def repair_package(path, document):
    metadata = (path / document['metadata_path']).read_bytes()
    index = (path / document['index_path']).read_bytes()
    document['metadata_sha256'] = hashlib.sha256(metadata).hexdigest()
    document['index_sha256'] = hashlib.sha256(index).hexdigest()
    document['root_hash'] = ''
    digest = hashlib.sha256(canonical(document) + metadata + index)
    for shard in document['shards']:
        digest.update(shard['sha256'].encode('ascii'))
    document['root_hash'] = digest.hexdigest()
    (path / 'qif.json').write_bytes(canonical(document))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', required=True)
    parser.add_argument('--rust-reader', required=True)
    parser.add_argument('--c-reader')
    args = parser.parse_args()
    checks = 0

    def commands(path, options=()):
        result = [[args.rust_reader, str(path), *options],
                  [sys.executable, str(Path(__file__).with_name('qif_reference_reader.py')), str(path), *options]]
        if args.c_reader:
            result.append([args.c_reader, str(path), *options])
        return result

    def check(path, accepted, options=()):
        nonlocal checks
        reports = []
        for command in commands(path, options):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            if accepted:
                if result.returncode:
                    raise AssertionError(f'{path.name}: {command[0]} failed: {result.stderr[-2000:]}')
                reports.append(json.loads(result.stdout))
            elif result.returncode == 0:
                raise AssertionError(f'{path.name}: {command[0]} accepted damaged/untrusted artifact')
            # A crash, sanitizer failure or timeout is not a clean rejection.
            elif result.returncode != 1 or 'Sanitizer' in result.stderr or 'Traceback (most recent call last)' in result.stderr:
                raise AssertionError(f'{path.name}: reader crashed: {result.stderr[-2000:]}')
        if accepted and any(report != reports[0] for report in reports[1:]):
            keys = [key for key in reports[0] if any(report[key] != reports[0][key] for report in reports[1:])]
            raise AssertionError(f'{path.name}: reports differ in {keys}')
        checks += 1

    with tempfile.TemporaryDirectory(prefix='qif-core-conformance-') as temporary:
        root = Path(temporary)
        fixtures = root / 'fixtures'
        subprocess.run([args.generator, str(fixtures)], check=True, timeout=30)
        key = (fixtures / 'trusted-key.hex').read_text()
        for codec in ('none', 'zstd', 'lz4'):
            for prefix, suffix in (('single', '.qif'), ('signed', '.qif'), ('package', '')):
                path = fixtures / (prefix + '-' + codec + suffix)
                check(path, True)
                check(path, prefix == 'signed', ('--trusted-key', key))
                check(path, False, ('--trusted-key', '11' * 32))
                if suffix:
                    relocated = root / ('reordered-' + path.name)
                    relocated.write_bytes(reordered(path.read_bytes()))
                    check(relocated, True)
        for name in ('empty.qif', 'empty-package', 'multi-chunk-zstd.qif', 'multi-chunk-lz4.qif'):
            check(fixtures / name, True)
        original = (fixtures / 'single-none.qif').read_bytes()
        for name, data in single_cases(original).items():
            path = root / (name + '.qif')
            path.write_bytes(data)
            check(path, False)
        # Signature failures include a rehashed control change, not just damage
        # detected by the container root. A valid self-signed key is not trust.
        original_signed = (fixtures / 'signed-zstd.qif').read_bytes()
        for name, position in [('signature', 44), ('public-key', 12), ('envelope-version', 4)]:
            data = bytearray(original_signed)
            data[u64(data, 120) + position] ^= 1
            path = root / (name + '.qif')
            path.write_bytes(data)
            check(path, False)
        data = bytearray(original_signed)
        data[u64(data, 40) + 12 + 8] ^= 1
        repair_single(data)
        path = root / 'signed-new-root.qif'
        path.write_bytes(data)
        check(path, False)

        # PACKAGE paths, topology, shard integrity, split blocks and aliases.
        for name in ('parent-path', 'absolute-path', 'symlink-escape', 'duplicate-shard',
                     'coordinate-rank', 'tensor-list', 'missing-shard', 'shard-size',
                     'shard-hash', 'index-shard-id', 'index-overlap', 'duplicate-json'):
            path = root / name
            shutil.copytree(fixtures / 'package-lz4', path)
            document = json.loads((path / 'qif.json').read_bytes())
            if name == 'parent-path':
                document['metadata_path'] = '../metadata.qmb'
            elif name == 'absolute-path':
                document['metadata_path'] = str(fixtures / 'package-lz4' / document['metadata_path'])
            elif name == 'symlink-escape':
                metadata_path = path / document['metadata_path']
                metadata_path.unlink()
                metadata_path.symlink_to(fixtures / 'package-lz4' / document['metadata_path'])
            elif name == 'duplicate-shard':
                document['shards'].append(copy.deepcopy(document['shards'][0]))
            elif name == 'coordinate-rank':
                document['shards'][0]['rank'] += 1
            elif name == 'tensor-list':
                document['shards'][0]['tensors'] = []
            elif name == 'missing-shard':
                (path / document['shards'][0]['path']).unlink()
            elif name == 'shard-size':
                document['shards'][0]['size'] += 1
            elif name == 'shard-hash':
                document['shards'][0]['sha256'] = '00' * 32
            elif name in ('index-shard-id', 'index-overlap'):
                index_path = path / document['index_path']
                index = bytearray(index_path.read_bytes())
                chunks = 48 + sum(u64(index, offset) for offset in (16, 24, 32))
                struct.pack_into('<I', index, chunks + (0 if name == 'index-shard-id' else 72),
                                 (1 << 32) - 1 if name == 'index-shard-id' else 0)
                index_path.write_bytes(index)
            if name == 'duplicate-json':
                data = (path / 'qif.json').read_bytes()
                (path / 'qif.json').write_bytes(b'{"schema_version":1,' + data[1:])
            elif name in ('parent-path', 'absolute-path', 'symlink-escape', 'missing-shard'):
                # Cannot repair inaccessible paths, but path validation must
                # happen before any attempt to use the external content.
                (path / 'qif.json').write_bytes(canonical(document))
            else:
                repair_package(path, document)
            check(path, False)
    readers = 'Rust/Python/C' if args.c_reader else 'Rust/Python'
    print(f'QIF core-profile {readers}: {checks} cases passed (reports, repaired negatives, trust policy)')


if __name__ == '__main__':
    main()
