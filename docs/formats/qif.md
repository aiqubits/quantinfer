# QIF format status

QIF version 1 is retired and is not the target format for new QuantInfer
artifacts. The historical wire description is preserved in
[qif-v1-legacy.md](qif-v1-legacy.md) only for migration and fixture analysis.

The new format is QIF v2.0. Its normative design is documented in
[qif-v2-final-plan.md](qif-v2-final-plan.md).

The v2 plan is a design contract, not an implementation claim. Until the v2
reader, writer, converters, and conformance fixtures are implemented, tools
must not silently label a v1 artifact as QIF v2. Legacy v1 reading and writing
must be explicitly isolated during migration and removed from the default
path once QIF v2 is available.
