"""The frozen ModelManifest schema; preserve unknown *requirements* as data."""
import json
import math
from pathlib import PurePosixPath

from .binary import U64_MAX, require

MODEL_KEYS = ('schema_version model_id architecture source_revision source_format '
              'tokenizer_revision tensors quantization execution_variants kernel_requirements '
              'checksums compatibility').split()
DESCRIPTOR_KEYS = ('storage_type dtype quantization block_size group_size scale_type '
                   'zero_point_type quantization_axis logical_shape physical_layout packing '
                   'backend_requirements').split()
PACKAGE_KEYS = ('schema_version profile model metadata_path metadata_sha256 index_path '
                'index_sha256 shards root_hash').split()
SHARD_KEYS = 'path size sha256 mesh_axis rank coordinate tensors'.split()
DTYPES = {'F32': (1, 1, 4, 'None', 'none'), 'F16': (2, 1, 2, 'None', 'none'),
          'BF16': (3, 1, 2, 'None', 'none'), 'Q4_0': (4, 32, 18, 'Q4_0', 'nibble_split_half'),
          'Q4_K': (5, 256, 144, 'Q4_K', 'k_quant'), 'Q5_K': (6, 256, 176, 'Q5_K', 'k_quant'),
          'Q6_K': (7, 256, 210, 'Q6_K', 'k_quant'), 'Q8_0': (8, 32, 34, 'Q8_0', 'scalar')}
SCHEMES = ['None', 'Q8_0', 'Q4_0', 'Q4_K', 'Q5_K', 'Q6_K', 'Mixed']


def require_object(value, keys):
    require(isinstance(value, dict) and list(value) == list(keys), 'noncanonical schema keys/order')


def uint(value, bits=64):
    require(type(value) is int and 0 <= value < 1 << bits, 'invalid unsigned integer')
    return value


def string(value, nonempty=False):
    require(isinstance(value, str) and (not nonempty or value), 'invalid string')
    require(len(value.encode('utf-8')) <= 16 * 1024 * 1024, 'string exceeds profile limit')
    return value


def strings(value):
    require(isinstance(value, list), 'expected string list')
    for entry in value:
        string(entry)


def checksum(value):
    require(isinstance(value, str) and len(value) == 64
            and all(ch in '0123456789abcdefABCDEF' for ch in value), 'invalid SHA-256')


def relative_path(value):
    string(value, True)
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and path.parts,
            'invalid relative package path')
    return path


def canonical(value):
    # The frozen schema contains integers and strings, no JSON floating values.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def load_json(data):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    # Bound depth before recursively constructing the JSON graph.
    depth, in_string, escaped = 0, False, False
    for char in data.decode('utf-8'):
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in '[{':
            depth += 1
            require(depth <= 32, 'JSON depth exceeds limit')
        elif char in ']}':
            depth -= 1
            require(depth >= 0, 'unbalanced JSON')
    require(depth == 0 and not in_string, 'unbalanced JSON')
    value = json.loads(data, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(canonical(value) == data, 'noncanonical JSON bytes')
    return value


def validate_model(model):
    require_object(model, MODEL_KEYS)
    require(uint(model['schema_version'], 32) == 1, 'unsupported manifest schema')
    for key in ('model_id', 'architecture', 'source_format'):
        string(model[key], True)
    for key in ('source_revision', 'tokenizer_revision'):
        if model[key] is not None:
            string(model[key])
    require(model['quantization'] in SCHEMES, 'unknown model quantization')
    tensors = model['tensors']
    require(isinstance(tensors, list) and len(tensors) <= 1_048_576, 'invalid tensor list')
    names, quantized = set(), False
    for tensor in tensors:
        require_object(tensor, ['name', 'descriptor', 'checksum_sha256'])
        name = string(tensor['name'], True)
        require(name not in names, 'duplicate tensor name')
        names.add(name)
        checksum(tensor['checksum_sha256'])
        descriptor = tensor['descriptor']
        require_object(descriptor, DESCRIPTOR_KEYS)
        dtype = descriptor['dtype']
        require(dtype in DTYPES, 'unknown dtype')
        _, block, _, scheme, packing = DTYPES[dtype]
        is_quantized = block != 1
        quantized |= is_quantized
        require(descriptor['storage_type'] == ('Quantized' if is_quantized else dtype)
                and descriptor['quantization'] == scheme, 'storage/dtype/scheme mismatch')
        require(descriptor['block_size'] == (block if is_quantized else None)
                and (descriptor['block_size'] is None or type(descriptor['block_size']) is int)
                and descriptor['group_size'] is None
                and descriptor['scale_type'] == ('F16' if is_quantized else None)
                and descriptor['zero_point_type'] == 'None'
                and descriptor['packing'] == packing, 'unsupported physical descriptor')
        shape = descriptor['logical_shape']
        require(isinstance(shape, list) and len(shape) <= 32, 'invalid shape/rank')
        for dimension in shape:
            require(uint(dimension) > 0, 'zero dimension')
        elements = math.prod(shape)
        require(elements <= U64_MAX and elements % block == 0, 'shape overflow/block mismatch')
        layout = descriptor['physical_layout']
        require(layout in ('row_major', 'ggml_output_rows'), 'unsupported layout')
        axis = descriptor['quantization_axis']
        require(axis == ('Tensor' if is_quantized else 'None') or
                (is_quantized and axis == {'Axis': 0} and type(axis['Axis']) is int
                 and layout == 'ggml_output_rows' and len(shape) == 2), 'unsupported quantization axis')
        strings(descriptor['backend_requirements'])
        if is_quantized:
            require(model['quantization'] in (scheme, 'Mixed'), 'tensor/model scheme mismatch')
    require(model['quantization'] == 'None' or quantized, 'quantized model has no quantized tensors')
    variants, variant_names = model['execution_variants'], set()
    require(isinstance(variants, list) and len(variants) <= 1_048_576, 'invalid variant list')
    for variant in variants:
        require_object(variant, ['name', 'quantization', 'dtype', 'runtime_provider', 'kernel_provider'])
        for key in ('name', 'runtime_provider', 'kernel_provider'):
            string(variant[key], True)
        string(variant['dtype'])
        require(variant['name'] not in variant_names and variant['quantization'] == model['quantization'],
                'duplicate variant/scheme mismatch')
        variant_names.add(variant['name'])
    kernel = model['kernel_requirements']
    require_object(kernel, ['operations', 'alignment', 'head_dimension', 'maximum_sequence_length'])
    strings(kernel['operations'])
    for key in ('alignment', 'head_dimension', 'maximum_sequence_length'):
        if kernel[key] is not None:
            uint(kernel[key])
    hashes = model['checksums']
    require(isinstance(hashes, dict) and list(hashes) == sorted(hashes), 'noncanonical checksum map')
    for path, digest in hashes.items():
        relative_path(path)
        checksum(digest)
    compatibility = model['compatibility']
    require_object(compatibility, ['minimum_quantinfer_version', 'gguf_version', 'required_features'])
    if compatibility['minimum_quantinfer_version'] is not None:
        string(compatibility['minimum_quantinfer_version'])
    if compatibility['gguf_version'] is not None:
        uint(compatibility['gguf_version'], 32)
    strings(compatibility['required_features'])


def validate_tensor(tensor, manifest):
    descriptor = manifest['descriptor']
    dtype, block, width, scheme, _ = DTYPES[descriptor['dtype']]
    require(tensor['dtype'] == tensor['physical_encoding'] == dtype
            and tensor['quantization_scheme'] == SCHEMES.index(scheme), 'wire encoding differs from manifest')
    require(tensor['shape'] == descriptor['logical_shape']
            and tensor['layout'] == (1 if descriptor['physical_layout'] == 'row_major' else 2),
            'wire shape/layout differs from manifest')
    require(tensor['raw_length'] == math.prod(tensor['shape']) // block * width,
            'tensor length differs from encoding')
    require(tensor['checksum_sha256'].lower() == manifest['checksum_sha256'].lower(),
            'tensor checksum differs from manifest')
