from __future__ import annotations

from typing import List, Tuple
from fpgai.ir.graph import Graph


def _dense_dims(op) -> Tuple[int, int]:
    in_f = int(op.attrs.get("in_features") or 0)
    out_f = int(op.attrs.get("out_features") or 0)
    if in_f <= 0 or out_f <= 0:
        raise ValueError(f"Dense {op.name} missing in/out features")
    return in_f, out_f


def _conv_dims(g: Graph, op) -> Tuple[int, int, int]:
    w_name = op.inputs[1] if len(op.inputs) > 1 else op.attrs.get("weight")
    if not w_name:
        raise ValueError(f"Conv {op.name} missing weight reference")

    w_t = g.get_tensor(w_name)
    if w_t is not None and getattr(w_t, "shape", None):
        w_shape = tuple(int(x) for x in w_t.shape)
    else:
        const = g.constants.get(w_name)
        if const is None or not hasattr(const, "shape"):
            raise ValueError(f"Conv {op.name} weight tensor {w_name} has no known shape")
        w_shape = tuple(int(x) for x in const.shape)

    if len(w_shape) != 4:
        raise ValueError(f"Conv {op.name} expected 4D weight shape, got {w_shape}")

    out_c, in_c, k1, k2 = w_shape
    if k1 != k2:
        raise ValueError(f"Conv {op.name} expected square kernel, got {w_shape}")
    return out_c, in_c, k1


def emit_weights_runtime_cpp(g: Graph) -> str:
    weighted_ops = [op for op in g.ops if op.op_type in ("Dense", "Conv")]

    lines: List[str] = []
    lines.append('#include "weights_runtime.h"')
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")
    lines.append("// Define weight/bias arrays expected by generated top/model code.")
    lines.append("// Arrays are emitted in weighted-op order: W0/B0, W1/B1, ...")
    lines.append("")

    # Definitions as flat arrays to match fpgai_params.h
    for idx, op in enumerate(weighted_ops):
        if op.op_type == "Conv":
            out_c, in_c, k = _conv_dims(g, op)
            total_w = out_c * in_c * k * k
            lines.append(f"// {op.name} (Conv)")
            lines.append(f"wgt_t W{idx}[{total_w}] = {{0}};")
            lines.append(f"bias_t B{idx}[{out_c}] = {{0}};")
            lines.append("")
        elif op.op_type == "Dense":
            in_f, out_f = _dense_dims(op)
            total_w = out_f * in_f
            lines.append(f"// {op.name} (Dense)")
            lines.append(f"wgt_t W{idx}[{total_w}] = {{0}};")
            lines.append(f"bias_t B{idx}[{out_f}] = {{0}};")
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
    lines.append("  // Expected order: weighted op order in the graph.")
    lines.append("")

    for idx, op in enumerate(weighted_ops):
        if op.op_type == "Conv":
            out_c, in_c, k = _conv_dims(g, op)
            total_w = out_c * in_c * k * k

            lines.append(f"  // Load {op.name} -> W{idx}[{total_w}] then B{idx}[{out_c}]")
            lines.append(f"  for (int i = 0; i < {total_w}; i++) {{")
            lines.append("    float v = _read_f32(inStream);")
            lines.append(f"    W{idx}[i] = (wgt_t)v;")
            lines.append("  }")
            lines.append(f"  for (int oc = 0; oc < {out_c}; oc++) {{")
            lines.append("    float v = _read_f32(inStream);")
            lines.append(f"    B{idx}[oc] = (bias_t)v;")
            lines.append("  }")
            lines.append("")

        elif op.op_type == "Dense":
            in_f, out_f = _dense_dims(op)
            total_w = out_f * in_f

            lines.append(f"  // Load {op.name} -> W{idx}[{total_w}] then B{idx}[{out_f}]")
            lines.append(f"  for (int i = 0; i < {total_w}; i++) {{")
            lines.append("    float v = _read_f32(inStream);")
            lines.append(f"    W{idx}[i] = (wgt_t)v;")
            lines.append("  }")
            lines.append(f"  for (int o = 0; o < {out_f}; o++) {{")
            lines.append("    float v = _read_f32(inStream);")
            lines.append(f"    B{idx}[o] = (bias_t)v;")
            lines.append("  }")
            lines.append("")

    lines.append("}")
    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)