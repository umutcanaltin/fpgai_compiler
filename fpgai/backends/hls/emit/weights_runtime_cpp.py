from __future__ import annotations


def emit_weights_runtime_cpp(graph) -> str:
    return '#include "weights_runtime.h"\n'