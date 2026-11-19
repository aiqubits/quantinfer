"""Independent two-chunk alignment fixtures; never change tensor logical values."""
import hashlib
import json
import struct

from qif_reference.codecs import decompress


def u64(data, offset):
    return struct.unpack_from('<Q', data, offset)[0]


def section(data, offset):
    start, length = struct.unpack_from('<QQ', data, offset)
    return data[start:start + length]


def aligned_single(original, alignment, uncompressed=False):
    """Relocate the unsigned singleton fixture, preserving its raw tensor hash."""
    assert alignment >= 4096 and alignment & (alignment - 1) == 0
    assert u64(original, 128) == 0
    index = bytearray(section(original, 56))
    assert struct.unpack_from('<II', index, 8) == (1, 2)
    record = 48 + u64(index, 16) + u64(index, 24)
    chunks = record + u64(index, 32)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    payload, raw = bytearray(), bytearray()
    stored_total = 0
    for i in range(2):
        chunk = chunks + i * 72
        codec = struct.unpack_from('<I', index, chunk + 4)[0]
        offset, size = struct.unpack_from('<QQ', index, chunk + 24)
        stored = section(original, 72)[offset:offset + size]
        assert hashlib.sha256(stored).digest() == index[chunk + 40:chunk + 72]
        decoded = decompress(codec, stored, u64(index, chunk + 16))
        assert u64(index, chunk + 8) == len(raw)
        raw.extend(decoded)
        if uncompressed:
            codec, stored = 0, decoded
        offset = align(len(payload))
        payload.extend(bytes(offset - len(payload)))
        payload.extend(stored)
        struct.pack_into('<I', index, chunk + 4, codec)
        struct.pack_into('<QQ', index, chunk + 24, offset, len(stored))
        index[chunk + 40:chunk + 72] = hashlib.sha256(stored).digest()
        stored_total += len(stored)
    assert len(raw) == u64(index, record + 48)
    assert hashlib.sha256(raw).digest() == index[record + 72:record + 104]
    struct.pack_into('<Q', index, record + 56, stored_total)
    if uncompressed:
        struct.pack_into('<I', index, record + 32, 0)
    controls = [section(original, 24), section(original, 40), bytes(index)]
    output = bytearray(original[:144])
    for field, data in zip((24, 40, 56, 72), controls + [payload]):
        offset = align(len(output))
        output.extend(bytes(offset - len(output)))
        output.extend(data)
        struct.pack_into('<QQ', output, field, offset, len(data))
    output[88:120] = hashlib.sha256(b''.join(controls)).digest()
    return output


def package_from_single(single, path):
    """Use identical tensor/chunk records and payload in one package shard."""
    path.mkdir()
    model = json.loads(section(single, 24))
    metadata, index, payload = (section(single, field) for field in (40, 56, 72))
    for name, data in [('metadata.qmb', metadata), ('index.qifi', index),
                       ('shard-00000.qifd', payload)]:
        (path / name).write_bytes(data)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    document = dict(schema_version=1, profile='QIF-PACKAGE', model=model,
                    metadata_path='metadata.qmb', metadata_sha256=digest(metadata),
                    index_path='index.qifi', index_sha256=digest(index),
                    shards=[dict(path='shard-00000.qifd', size=len(payload),
                                 sha256=digest(payload), mesh_axis=0, rank=1,
                                 coordinate=[0], tensors=['weight'])], root_hash='')
    return document
