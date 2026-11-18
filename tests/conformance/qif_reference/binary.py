"""Little-endian cursors, QMB metadata and the fixed QIDX tables."""
import hashlib
import math
import struct

LIMIT = 64 * 1024 * 1024
U64_MAX = (1 << 64) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def region(data, offset, length):
    require(0 <= offset <= len(data) and 0 <= length <= len(data) - offset,
            'truncated or overflowing range')
    return data[offset:offset + length]


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Cursor:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def read(self, length):
        value = region(self.data, self.offset, length)
        self.offset += length
        return value

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self, fmt='Q'):
        length = self.number(fmt)
        require(length <= 16 * 1024 * 1024, 'string exceeds profile limit')
        return self.read(length).decode('utf-8')

    def end(self):
        require(self.offset == len(self.data), 'trailing bytes')


def metadata(data):
    cursor = Cursor(data)

    def value(depth):
        require(depth <= 32, 'QMB nesting exceeds limit')
        tag = cursor.number('I')
        numeric = {4: ('U32', 'I'), 5: ('I32', 'i'), 6: ('F32', 'f'),
                   10: ('U64', 'Q'), 11: ('I64', 'q'), 12: ('F64', 'd')}
        if tag in numeric:
            name, fmt = numeric[tag]
            raw = cursor.read(struct.calcsize('<' + fmt))
            number = struct.unpack('<' + fmt, raw)[0]
            if tag in (6, 12):
                require(math.isfinite(number), 'non-finite metadata')
                # Compare IEEE bits, without decimal float formatting differences.
                number = raw.hex()
            return {'type': name, 'value': number}
        if tag == 7:
            number = cursor.number('B')
            require(number in (0, 1), 'invalid QMB boolean')
            return {'type': 'Bool', 'value': bool(number)}
        if tag == 8:
            return {'type': 'String', 'value': cursor.string()}
        require(tag in (9, 13), 'unknown QMB type')
        count = cursor.number('Q')
        require(count <= 1_048_576, 'QMB item count exceeds limit')
        values = [] if tag == 9 else {}
        previous = None
        for _ in range(count):
            if tag == 13:
                key = cursor.string()
                encoded = key.encode('utf-8')
                require(key and (previous is None or previous < encoded),
                        'QMB keys must be sorted, nonempty and unique')
                previous = encoded
                values[key] = value(depth + 1)
            else:
                entry = value(depth + 1)
                require(not values or entry['type'] == values[0]['type'],
                        'QMB arrays must be homogeneous')
                values.append(entry)
        return {'type': 'Array' if tag == 9 else 'Map', 'value': values}

    result = value(0)
    require(result['type'] == 'Map', 'QMB root must be a map')
    cursor.end()
    return result['value']


def index(data):
    cursor = Cursor(data)
    require(cursor.read(4) == b'QIDX' and cursor.number('I') == 1, 'invalid QIDX header')
    tensor_count, chunk_count = cursor.number('I'), cursor.number('I')
    require(tensor_count <= 1_048_576 and chunk_count <= 4_194_304, 'QIDX count exceeds limit')
    lengths = [cursor.number('Q') for _ in range(4)]
    require(lengths[2:] == [tensor_count * 120, chunk_count * 72], 'QIDX record sizes differ')
    strings, shapes, records, chunk_bytes = [cursor.read(n) for n in lengths]
    cursor.end()
    tensors, chunks = [], []
    for n in range(tensor_count):
        record = Cursor(region(records, n * 120, 120))
        tensor_id = record.number('Q')
        name_ref, shape_ref = record.number('I'), record.number('I')
        name_cursor = Cursor(region(strings, name_ref, len(strings) - name_ref))
        name = name_cursor.string('I')
        shape_cursor = Cursor(region(shapes, shape_ref, len(shapes) - shape_ref))
        rank = shape_cursor.number('I')
        require(rank <= 32, 'tensor rank exceeds limit')
        shape = [shape_cursor.number('Q') for _ in range(rank)]
        dtype, encoding, scheme, layout, codec = [record.number('I') for _ in range(5)]
        flags, start, count = [record.number('I') for _ in range(3)]
        raw, stored, offset = [record.number('Q') for _ in range(3)]
        checksum, alias = record.read(32).hex(), record.number('Q')
        require(flags == record.number('I') == record.number('I') == 0,
                'tensor flags/reserved fields must be zero')
        require(tensor_id == n and start + count <= chunk_count, 'invalid tensor id/chunk range')
        tensors.append(dict(tensor_id=tensor_id, name=name, shape=shape, dtype=dtype,
                            physical_encoding=encoding, quantization_scheme=scheme,
                            layout=layout, codec=codec, raw_length=raw, stored_length=stored,
                            data_offset=offset, checksum_sha256=checksum,
                            alias_of=None if alias == U64_MAX else alias,
                            chunk_start=start, chunk_count=count))
    for n in range(chunk_count):
        record = Cursor(region(chunk_bytes, n * 72, 72))
        shard, codec = record.number('I'), record.number('I')
        raw_offset, raw, stored_offset, stored = [record.number('Q') for _ in range(4)]
        require(codec in (0, 1, 2) and raw > 0 and stored > 0, 'invalid chunk codec/length')
        require(raw_offset + raw <= U64_MAX and stored_offset + stored <= U64_MAX,
                'chunk range overflows')
        require(stored_offset % 4096 == 0, 'unaligned chunk')
        require((codec == 0 and raw == stored) or
                (codec != 0 and raw <= 128 * 1024 * 1024 and stored <= 129 * 1024 * 1024),
                'chunk resource profile mismatch')
        chunks.append(dict(shard_id=shard, codec=codec, raw_offset=raw_offset,
                           raw_length=raw, stored_offset=stored_offset, stored_length=stored,
                           checksum_sha256=record.read(32).hex()))
    return tensors, chunks


def finite_payload(dtype, data):
    width, exponent, stride, offsets = {
        1: (4, 0x7f800000, 4, [0]), 2: (2, 0x7c00, 2, [0]),
        3: (2, 0x7f80, 2, [0]), 4: (2, 0x7c00, 18, [0]),
        5: (2, 0x7c00, 144, [0, 2]), 6: (2, 0x7c00, 176, [0, 2]),
        7: (2, 0x7c00, 210, [208]), 8: (2, 0x7c00, 34, [0]),
    }[dtype]
    require(len(data) % stride == 0, 'incomplete physical block')
    for start in range(0, len(data), stride):
        for offset in offsets:
            bits = int.from_bytes(data[start + offset:start + offset + width], 'little')
            require(bits & exponent != exponent, 'non-finite tensor value/scale')
