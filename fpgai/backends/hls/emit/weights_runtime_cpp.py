from __future__ import annotations

from typing import List
from fpgai.ir.graph import Graph


def emit_weights_runtime_cpp(g: Graph) -> str:
    dense_ops = [op for op in g.ops if op.op_type == "Dense"]

    lines: List[str] = []
    lines.append('#include "weights_runtime.h"')
    lines.append("")
    lines.append("// Define weight/bias arrays (stream-injected at runtime)")
    lines.append("")

    # Define arrays
    for op in dense_ops:
        name = op.name
        in_f = int(op.attrs.get("in_features") or 0)
        out_f = int(op.attrs.get("out_features") or 0)
        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"Dense {name} missing in/out features")

        lines.append(f"wgt_t W_{name}[{out_f}][{in_f}] = {{0}};")
        lines.append(f"wgt_t B_{name}[{out_f}] = {{0}};")
        lines.append("")

    lines.append("static inline float _read_f32(hls::stream<axis32_t>& s) {")
    lines.append("#pragma HLS INLINE")
    lines.append("  union { unsigned int u; float f; } cvt;")
    lines.append("  axis32_t pkt = s.read();")
    lines.append("  cvt.u = (unsigned int)pkt.data;")
    lines.append("  return cvt.f;")
    lines.append("}")
    lines.append("")
    lines.append("void fpgai_load_weights_from_stream(hls::stream<axis32_t>& inStream) {")
    lines.append("#pragma HLS INLINE off")
    lines.append("  // Expected order: for each Dense layer: W[out][in] then B[out]")
    lines.append("")

    for op in dense_ops:
        name = op.name
        in_f = int(op.attrs.get("in_features") or 0)
        out_f = int(op.attrs.get("out_features") or 0)

        lines.append(f"  // Load {name}: W[{out_f}][{in_f}] then B[{out_f}]")
        lines.append(f"  for (int o = 0; o < {out_f}; o++) {{")
        lines.append(f"    for (int i = 0; i < {in_f}; i++) {{")
        lines.append("      float v = _read_f32(inStream);")
        lines.append(f"      W_{name}[o][i] = (wgt_t)v;")
        lines.append("    }")
        lines.append("  }")
        lines.append(f"  for (int o = 0; o < {out_f}; o++) {{")
        lines.append("    float v = _read_f32(inStream);")
        lines.append(f"    B_{name}[o] = (wgt_t)v;")
        lines.append("  }")
        lines.append("")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
