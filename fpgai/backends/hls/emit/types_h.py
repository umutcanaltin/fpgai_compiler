from __future__ import annotations

from fpgai.ir.graph import Graph


def _spec_to_ap(spec) -> str:
    if not isinstance(spec, dict):
        return "ap_fixed<16,6>"
    tb = int(spec.get("total_bits", 16))
    ib = int(spec.get("int_bits", 6))
    return f"ap_fixed<{tb},{ib}>"


def _default_precision():
    return {
        "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    }


def emit_types_h(graph: Graph, *, top_name: str) -> str:
    dflt = _default_precision()

    lines = []
    lines.append("#pragma once")
    lines.append("#include <ap_fixed.h>")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    lines.append("// Global fallback typedefs")
    lines.append(f"typedef {_spec_to_ap(dflt['activation'])} act_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['weight'])} wgt_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['bias'])} bias_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['accum'])} acc_t;")
    lines.append("")

    for idx, op in enumerate(graph.ops):
        p = op.attrs.get("precision", dflt)
        tag = op.attrs.get("precision_tag", f"op{idx}")

        lines.append(f"// {idx}: {op.name} ({op.op_type})")
        lines.append(f"typedef {_spec_to_ap(p['activation'])} {tag}_act_t;")
        lines.append(f"typedef {_spec_to_ap(p['weight'])} {tag}_wgt_t;")
        lines.append(f"typedef {_spec_to_ap(p['bias'])} {tag}_bias_t;")
        lines.append(f"typedef {_spec_to_ap(p['accum'])} {tag}_acc_t;")
        lines.append("")

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)