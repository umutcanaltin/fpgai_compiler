from __future__ import annotations

from typing import List
import numpy as np


def _numel_from_graph(graph, tensor_name: str) -> int:
    if hasattr(graph, "constants") and tensor_name in graph.constants:
        return int(np.asarray(graph.constants[tensor_name]).size)

    if hasattr(graph, "params") and tensor_name in graph.params:
        return int(np.asarray(graph.params[tensor_name]).size)

    try:
        t = graph.get_tensor(tensor_name)
    except Exception:
        t = None

    if t is not None:
        shape = getattr(t, "shape", None)
        if shape is not None:
            n = 1
            for d in shape:
                n *= int(d)
            return int(n)
        data = getattr(t, "data", None)
        if data is not None:
            return int(np.asarray(data).size)

    return 0


def emit_weights_runtime_h(graph) -> str:
    lines: List[str] = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append("#include <ap_int.h>")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")
    lines.append("static inline wgt_t fpga_bits_to_wgt(ap_uint<32> bits) {")
    lines.append("    union { unsigned int i; float f; } c;")
    lines.append("    c.i = bits.to_uint();")
    lines.append("    return (wgt_t)c.f;")
    lines.append("}")
    lines.append("")
    lines.append("static inline bias_t fpga_bits_to_bias(ap_uint<32> bits) {")
    lines.append("    union { unsigned int i; float f; } c;")
    lines.append("    c.i = bits.to_uint();")
    lines.append("    return (bias_t)c.f;")
    lines.append("}")
    lines.append("")
    lines.append("template<int N>")
    lines.append("void load_wgt_vector(const ap_uint<32>* weights_mem, int base, wgt_t out[N]) {")
    lines.append("#pragma HLS INLINE off")
    lines.append("    for (int i = 0; i < N; ++i) {")
    lines.append("#pragma HLS PIPELINE II=1")
    lines.append("        out[i] = fpga_bits_to_wgt(weights_mem[base + i]);")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("template<int N>")
    lines.append("void load_bias_vector(const ap_uint<32>* weights_mem, int base, bias_t out[N]) {")
    lines.append("#pragma HLS INLINE off")
    lines.append("    for (int i = 0; i < N; ++i) {")
    lines.append("#pragma HLS PIPELINE II=1")
    lines.append("        out[i] = fpga_bits_to_bias(weights_mem[base + i]);")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)