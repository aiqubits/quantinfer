"""Independent decoding: raw LZ4 blocks and the system libzstd API."""
import ctypes as c
import ctypes.util

from .binary import Cursor, LIMIT, require


def decompress(codec, data, expected):
    # This verifier additionally bounds each reconstructed tensor/artifact.
    require(expected <= LIMIT, 'reference reader allocation budget exceeded')
    if codec == 0:
        result = data
    elif codec == 2:
        cursor, output = Cursor(data), bytearray()
        require(cursor.number('I') == expected, 'LZ4 size prefix differs')

        def length(initial):
            total = initial
            if initial == 15:
                while True:
                    extra = cursor.number('B')
                    total += extra
                    require(total <= expected, 'LZ4 sequence exceeds output')
                    if extra != 255:
                        break
            return total

        while cursor.offset < len(data):
            token = cursor.number('B')
            literals = length(token >> 4)
            require(literals <= expected - len(output), 'LZ4 literals exceed output')
            output.extend(cursor.read(literals))
            if cursor.offset == len(data):
                break
            distance = cursor.number('H')
            require(0 < distance <= len(output), 'invalid LZ4 match distance')
            matched = length(token & 15) + 4
            require(matched <= expected - len(output), 'LZ4 match exceeds output')
            for _ in range(matched):
                output.append(output[-distance])
        result = bytes(output)
    else:
        require(codec == 1, 'unknown codec')
        lib = c.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
        lib.ZSTD_createDCtx.restype = c.c_void_p
        lib.ZSTD_freeDCtx.argtypes = [c.c_void_p]
        lib.ZSTD_DCtx_setParameter.argtypes = [c.c_void_p, c.c_int, c.c_int]
        lib.ZSTD_DCtx_setParameter.restype = c.c_size_t
        class InBuffer(c.Structure):
            _fields_ = [('src', c.c_void_p), ('size', c.c_size_t), ('pos', c.c_size_t)]

        class OutBuffer(c.Structure):
            _fields_ = [('dst', c.c_void_p), ('size', c.c_size_t), ('pos', c.c_size_t)]

        lib.ZSTD_decompressStream.argtypes = [c.c_void_p, c.POINTER(OutBuffer), c.POINTER(InBuffer)]
        lib.ZSTD_decompressStream.restype = c.c_size_t
        lib.ZSTD_isError.argtypes = [c.c_size_t]
        lib.ZSTD_isError.restype = c.c_uint
        context = lib.ZSTD_createDCtx()
        require(context, 'cannot allocate zstd context')
        try:
            # ZSTD_d_windowLogMax = 100, a stable public API parameter.
            status = lib.ZSTD_DCtx_setParameter(context, 100, 27)
            require(not lib.ZSTD_isError(status), 'cannot bound zstd window')
            source = c.create_string_buffer(data)
            output = c.create_string_buffer(expected + 1)
            incoming = InBuffer(c.cast(source, c.c_void_p), len(data), 0)
            outgoing = OutBuffer(c.cast(output, c.c_void_p), expected + 1, 0)
            while True:
                before = (incoming.pos, outgoing.pos)
                status = lib.ZSTD_decompressStream(context, c.byref(outgoing), c.byref(incoming))
                require(not lib.ZSTD_isError(status) and outgoing.pos <= expected,
                        'invalid zstd frame/output length')
                if status == 0 and incoming.pos == len(data):
                    break
                require(before != (incoming.pos, outgoing.pos), 'truncated zstd frame')
            require(outgoing.pos == expected, 'zstd output length differs')
            result = output.raw[:expected]
        finally:
            lib.ZSTD_freeDCtx(context)
    require(len(result) == expected, 'decoded length differs')
    return result
