"""Independent, bounded QIF wire validator used by conformance tests.

This package never imports QuantInfer or calls its runtime/bindings. Its local
64 MiB artifact budget is deliberately smaller than the production profile.
"""
