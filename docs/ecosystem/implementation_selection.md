# Explainable implementation selection

Selection first filters candidates by execution mode, backend, language, board, toolchain, precision, interfaces, memory assumptions, and minimum validation level. Explicit compatible preferences win. Automatic policies are deterministic and report every accepted or rejected candidate.

Supported policies are `explicit_only`, `validated_only`, `latency`, `throughput`, `area`, `power`, and `balanced`.
