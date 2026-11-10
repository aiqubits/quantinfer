"""Python bindings for the QuantInfer CPU runtime."""

import ctypes as c
import operator
import os
from threading import RLock
from weakref import finalize

from ._ffi import GenerationOptions, QuantInferError, check, load_library

__all__ = ["EOS_AUTO", "Model", "QuantInferError"]

EOS_AUTO = (1 << 32) - 1
_EOS_DISABLED = EOS_AUTO - 1
_SIZE_MAX = (1 << (8 * c.sizeof(c.c_size_t))) - 1


def _unsigned(value, maximum, name):
    value = operator.index(value)
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside its unsigned integer range")
    return value


def _utf8(text):
    if "\0" in text:
        raise ValueError("input strings cannot contain NUL")
    return text.encode("utf-8")


def _tokens(values):
    values = [_unsigned(value, EOS_AUTO, "token ID") for value in values]
    return (c.c_uint32 * len(values))(*values)


class Model:
    """A loaded GGUF file, QIF file, or verified QuantInfer model package."""

    def __init__(self, path, *, library=None):
        self._lock = RLock()
        self._native = load_library(library)
        self._handle = c.c_void_p()
        status = self._native.quantinfer_model_open(_utf8(os.fsdecode(path)), c.byref(self._handle))
        check(self._native, status)
        self._release = finalize(self, self._native.quantinfer_model_free, self._handle)

    def close(self):
        """Release the model once all calls on this Python handle finish."""
        with self._lock:
            self._release()

    def __enter__(self):
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_open(self):
        if not self._release.alive:
            raise QuantInferError("model is closed")

    def _read_output(self, call, element):
        count = c.c_size_t()
        check(self._native, call(None, 0, c.byref(count)), allow_size_query=True)
        output = (element * count.value)()
        check(self._native, call(output, len(output), c.byref(count)))
        return output, count.value

    def tokenize(self, text):
        encoded = _utf8(text)
        with self._lock:
            self._require_open()
            output, count = self._read_output(
                lambda out, size, written: self._native.quantinfer_tokenize(
                    self._handle, encoded, out, size, written
                ),
                c.c_uint32,
            )
            return list(output[:count])

    def detokenize(self, tokens):
        values = _tokens(tokens)
        with self._lock:
            self._require_open()
            output, count = self._read_output(
                lambda out, size, written: self._native.quantinfer_detokenize(
                    self._handle, values, len(values), out, size, written
                ),
                c.c_char,
            )
            return c.string_at(output, count - 1).decode("utf-8")

    def generate_tokens(
        self,
        tokens,
        *,
        max_new_tokens=32,
        context_length=None,
        temperature=0.0,
        top_k=None,
        top_p=None,
        eos_token_id=EOS_AUTO,
        seed=0,
    ):
        """Generate token IDs. The default uses the model EOS and None disables it."""
        values = _tokens(tokens)
        options = GenerationOptions(
            0 if context_length is None
            else _unsigned(context_length, _SIZE_MAX, "context_length"),
            _unsigned(max_new_tokens, _SIZE_MAX, "max_new_tokens"),
            temperature,
            0 if top_k is None else _unsigned(top_k, _SIZE_MAX, "top_k"),
            0.0 if top_p is None else top_p,
            _EOS_DISABLED if eos_token_id is None
            else _unsigned(eos_token_id, EOS_AUTO, "EOS token ID"),
            _unsigned(seed, (1 << 64) - 1, "seed"),
        )
        with self._lock:
            self._require_open()
            output, count = self._read_output(
                lambda out, size, written: self._native.quantinfer_generate_tokens(
                    self._handle, values, len(values), c.byref(options), out, size, written
                ),
                c.c_uint32,
            )
            return list(output[:count])

    def generate(self, prompt, **options):
        """Encode a prompt and return its generated continuation as text."""
        with self._lock:
            return self.detokenize(self.generate_tokens(self.tokenize(prompt), **options))
