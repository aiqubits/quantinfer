"""QIF-SINGLE/PACKAGE independent validation and reproducible wire reports."""
import hashlib
from pathlib import Path
import subprocess
import tempfile

from .binary import Cursor, LIMIT, U64_MAX, finite_payload, index, metadata, region, require, sha
from .codecs import decompress
from .schema import (PACKAGE_KEYS, SHARD_KEYS, canonical, checksum, load_json,
                     relative_path, require_object, string, strings, uint,
                     validate_model, validate_tensor)


def read_file(path, maximum=LIMIT):
    with path.open('rb') as source:
        data = source.read(maximum + 1)
    require(len(data) <= maximum, 'reference reader file budget exceeded')
    return data


def resolved(root, relative):
    relative_path(relative)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), 'package path escapes root')
    return path


def verify_signature(root_hash, envelope, trusted_keys):
    if envelope is None:
        require(not trusted_keys, 'a trusted signature is required')
        return None
    cursor = Cursor(envelope)
    require(cursor.read(4) == b'QSIG' and cursor.number('I') == cursor.number('I') == 1,
            'invalid signature envelope')
    key, signature = cursor.read(32), cursor.read(64)
    cursor.end()
    # OpenSSL's Ed25519 implementation is independent of ed25519-dalek.
    with tempfile.TemporaryDirectory(prefix='qif-signature-') as directory:
        directory = Path(directory)
        (directory / 'key.der').write_bytes(bytes.fromhex('302a300506032b6570032100') + key)
        (directory / 'signature').write_bytes(signature)
        (directory / 'message').write_bytes(bytes.fromhex(root_hash))
        result = subprocess.run(['openssl', 'pkeyutl', '-verify', '-pubin', '-keyform', 'DER',
                                 '-inkey', str(directory / 'key.der'), '-rawin',
                                 '-in', str(directory / 'message'),
                                 '-sigfile', str(directory / 'signature')],
                                capture_output=True, timeout=10, check=False)
        require(result.returncode == 0, 'Ed25519 verification failed')
    require(not trusted_keys or key.hex() in trusted_keys, 'untrusted signing key')
    return dict(public_key=key.hex(), verified=True,
                publisher_trust='trusted' if trusted_keys else 'not_evaluated')


def single(path, trusted_keys):
    data = read_file(path)
    header = Cursor(region(data, 0, 144))
    require(header.read(4) == b'QIF\0', 'invalid QIF magic')
    require([header.number('I') for _ in range(5)] == [2, 0, 144, 1, 1],
            'unsupported wire version/flags/hash/index')
    bounds = [(header.number('Q'), header.number('Q')) for _ in range(4)]
    root_hash = header.read(32).hex()
    signature_offset, signature_length, reserved = [header.number('Q') for _ in range(3)]
    require(reserved == 0, 'nonzero header reserved field')
    ranges = []
    sections = []
    for start, length in bounds:
        require(start >= 144 and start % 4096 == 0, 'unaligned/overlapping header range')
        sections.append(region(data, start, length))
        ranges.append((start, start + length))
    require(sum(bounds[3]) == len(data), 'unexpected bytes outside payload')
    if signature_length == 0:
        require(signature_offset == 0, 'nonzero unsigned signature offset')
        envelope = None
    else:
        require(signature_length == 108 and signature_offset >= 144
                and signature_offset % 4096 == 0, 'invalid signature range')
        envelope = region(data, signature_offset, signature_length)
        ranges.append((signature_offset, signature_offset + signature_length))
    ranges.sort()
    require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])),
            'overlapping sections')
    manifest_bytes, metadata_bytes, index_bytes, payload = sections
    require(sha(manifest_bytes + metadata_bytes + index_bytes) == root_hash, 'root hash mismatch')
    model = load_json(manifest_bytes)
    validate_model(model)
    signature = verify_signature(root_hash, envelope, trusted_keys)
    return model, metadata_bytes, index_bytes, {0: payload}, root_hash, signature


def package(path, trusted_keys):
    require(not trusted_keys, 'PACKAGE signatures are outside this profile')
    root = path.resolve(strict=True)
    package_bytes = read_file(resolved(root, 'qif.json'))
    document = load_json(package_bytes)
    require_object(document, PACKAGE_KEYS)
    require(uint(document['schema_version'], 32) == 1
            and document['profile'] in ('QIF-PACKAGE', 'QIF2-PACKAGE'), 'unsupported package profile')
    validate_model(document['model'])
    metadata_bytes = read_file(resolved(root, document['metadata_path']))
    index_bytes = read_file(resolved(root, document['index_path']))
    require(sha(metadata_bytes) == document['metadata_sha256']
            and sha(index_bytes) == document['index_sha256'], 'package control checksum mismatch')
    require(isinstance(document['shards'], list), 'invalid shard list')
    unsigned = dict(document, root_hash='')
    digest = hashlib.sha256(canonical(unsigned) + metadata_bytes + index_bytes)
    for shard in document['shards']:
        require_object(shard, SHARD_KEYS)
        checksum(shard['sha256'])
        digest.update(shard['sha256'].encode('ascii'))
    require(digest.hexdigest() == document['root_hash'], 'package root hash mismatch')
    shards, names, budget = {}, set(), len(package_bytes) + len(metadata_bytes) + len(index_bytes)
    require(budget <= LIMIT, 'reference artifact allocation budget exceeded')
    for shard in document['shards']:
        name = string(shard['path'], True)
        require(name not in names, 'duplicate shard path')
        names.add(name)
        budget += uint(shard['size'])
        require(budget <= LIMIT, 'reference artifact allocation budget exceeded')
        data = read_file(resolved(root, name), shard['size'])
        require(len(data) == shard['size'] and sha(data) == shard['sha256'], 'shard checksum/size mismatch')
        uint(shard['mesh_axis'], 32)
        rank = uint(shard['rank'], 32)
        coordinate = shard['coordinate']
        require(rank > 0 and isinstance(coordinate, list) and len(coordinate) == rank,
                'invalid mesh coordinate')
        for position in coordinate:
            uint(position, 32)
        strings(shard['tensors'])
        shards[name] = data
    return document, metadata_bytes, index_bytes, shards


def report(path, trusted_keys=()):
    path = Path(path)
    if path.is_dir():
        document, metadata_bytes, index_bytes, named_shards = package(path, trusted_keys)
        model, root_hash, signature, profile = document['model'], document['root_hash'], None, 'PACKAGE'
        # The global index names shard-NNNNN.qifd by ID. Other listed shards
        # remain hash-verified even if no tensor references them.
        shards = named_shards
    else:
        model, metadata_bytes, index_bytes, shards, root_hash, signature = single(path, trusted_keys)
        document, profile = None, 'SINGLE'
    decoded_metadata = metadata(metadata_bytes)
    tensors, chunks = index(index_bytes)
    require(len(tensors) == len(model['tensors']), 'manifest/index tensor count differs')
    by_name = {tensor['name']: tensor for tensor in model['tensors']}
    names, referenced, occupied, payloads = set(), set(), {}, []
    reconstructed_budget = 0
    for tensor in tensors:
        name = tensor['name']
        require(name in by_name and name not in names, 'unknown/duplicate index tensor')
        names.add(name)
        validate_tensor(tensor, by_name[name])
        start, count, alias = tensor['chunk_start'], tensor['chunk_count'], tensor['alias_of']
        if alias is not None:
            require(alias < tensor['tensor_id'] and count == 0, 'invalid alias target/chunks')
            target = tensors[alias]
            require(target['alias_of'] is None and target['chunk_count'] > 0, 'alias target must be physical')
            for key in ('shape', 'dtype', 'physical_encoding', 'quantization_scheme', 'layout', 'codec',
                        'raw_length', 'stored_length', 'checksum_sha256', 'data_offset'):
                require(tensor[key] == target[key], 'alias metadata differs')
            payloads.append(payloads[alias])
            continue
        require(count > 0, 'physical tensor has no chunks')
        reconstructed_budget += tensor['raw_length']
        require(reconstructed_budget <= LIMIT, 'reference decoded tensor budget exceeded')
        raw_total, stored_total, ends, decoded = 0, 0, {}, bytearray()
        for position in range(start, start + count):
            require(position not in referenced, 'chunk reused by multiple physical tensors')
            referenced.add(position)
            chunk = chunks[position]
            shard_id = chunk['shard_id']
            shard_key = f'shard-{shard_id:05}.qifd' if document else shard_id
            require((document is not None or shard_id == 0) and shard_key in shards, 'missing shard')
            offset, length = chunk['stored_offset'], chunk['stored_length']
            if position == start:
                require(offset == tensor['data_offset'], 'tensor data offset differs')
            if shard_id in ends:
                require(offset == (ends[shard_id] + 4095) // 4096 * 4096, 'chunk alignment sequence differs')
            ends[shard_id] = offset + length
            require(chunk['codec'] == tensor['codec'] and chunk['raw_offset'] == raw_total,
                    'chunk codec/raw offset differs')
            stored = region(shards[shard_key], offset, length)
            require(sha(stored) == chunk['checksum_sha256'], 'stored chunk checksum mismatch')
            decoded.extend(decompress(chunk['codec'], stored, chunk['raw_length']))
            raw_total += chunk['raw_length']
            stored_total += length
            occupied.setdefault(shard_key, []).append((offset, offset + length))
        require(raw_total == tensor['raw_length'] and stored_total == tensor['stored_length'],
                'tensor/chunk lengths differ')
        require(raw_total <= U64_MAX and stored_total <= U64_MAX, 'tensor lengths overflow')
        require(sha(decoded) == tensor['checksum_sha256'], 'raw tensor checksum mismatch')
        finite_payload(tensor['dtype'], decoded)
        payloads.append(bytes(decoded))
    require(len(referenced) == len(chunks), 'unreferenced chunk')
    for ranges in occupied.values():
        ranges.sort()
        require(all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), 'overlapping chunks')
    if document:
        indexed = {}
        for tensor in tensors:
            physical = tensor if tensor['alias_of'] is None else tensors[tensor['alias_of']]
            for chunk in chunks[physical['chunk_start']:physical['chunk_start'] + physical['chunk_count']]:
                indexed.setdefault(f"shard-{chunk['shard_id']:05}.qifd", set()).add(tensor['name'])
        for shard in document['shards']:
            require(set(shard['tensors']) == indexed.get(shard['path'], set()), 'shard tensor map differs')
    for tensor, payload in zip(tensors, payloads):
        tensor['payload_hex'] = payload.hex()
    return dict(profile=profile, root_hash=root_hash, model=model, metadata=decoded_metadata,
                metadata_sha256=sha(metadata_bytes), index_sha256=sha(index_bytes),
                tensors=tensors, chunks=chunks, signature=signature)
