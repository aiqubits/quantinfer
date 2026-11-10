import ctypes as c
from ctypes.util import find_library
import os
from pathlib import Path
import sys


class QuantInferError(RuntimeError):
    """A failure reported by the native QuantInfer runtime."""


class GenerationOptions(c.Structure):
    _fields_ = [
        ("context_length", c.c_size_t),
        ("max_new_tokens", c.c_size_t),
        ("temperature", c.c_float),
        ("top_k", c.c_size_t),
        ("top_p", c.c_float),
        ("eos_token_id", c.c_uint32),
        ("seed", c.c_uint64),
    ]


def load_library(path):
    path = path if path is not None else os.environ.get("QUANTINFER_LIBRARY")
    if path is not None:
        name = str(Path(path).expanduser().resolve())
    else:
        name = find_library("quantinfer_c")
        if not name:
            if sys.platform == "win32":
                name = "quantinfer_c.dll"
            elif sys.platform == "darwin":
                name = "libquantinfer_c.dylib"
            else:
                name = "libquantinfer_c.so"
    try:
        library = c.CDLL(name)
    except OSError as error:
        raise QuantInferError(
            f"cannot load QuantInfer C library {name}. Set QUANTINFER_LIBRARY to its path"
        ) from error

    version = _function(library, "quantinfer_abi_version", c.c_uint32, [])
    if version() != 1:
        raise QuantInferError("QuantInfer C library has an unsupported ABI version")
    tokens = c.POINTER(c.c_uint32)
    count = c.POINTER(c.c_size_t)
    signatures = {
        "quantinfer_last_error": (c.c_char_p, []),
        "quantinfer_model_open": (c.c_int32, [c.c_char_p, c.POINTER(c.c_void_p)]),
        "quantinfer_model_free": (None, [c.c_void_p]),
        "quantinfer_tokenize": (c.c_int32, [c.c_void_p, c.c_char_p, tokens, c.c_size_t, count]),
        "quantinfer_detokenize": (
            c.c_int32, [c.c_void_p, tokens, c.c_size_t, c.POINTER(c.c_char), c.c_size_t, count],
        ),
        "quantinfer_generate_tokens": (
            c.c_int32,
            [c.c_void_p, tokens, c.c_size_t, c.POINTER(GenerationOptions), tokens, c.c_size_t, count],
        ),
    }
    for name, (result, arguments) in signatures.items():
        _function(library, name, result, arguments)
    return library


def _function(library, name, result, arguments):
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise QuantInferError(f"QuantInfer C library is missing {name}") from error
    function.argtypes = arguments
    function.restype = result
    return function


def check(library, status, allow_size_query=False):
    if status == 0 or (allow_size_query and status == 2):
        return
    raw = library.quantinfer_last_error()
    message = raw.decode("utf-8", errors="replace") if raw else "QuantInfer operation failed"
    if status == 1:
        raise ValueError(message)
    raise QuantInferError(message)
