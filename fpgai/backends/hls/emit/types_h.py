from __future__ import annotations

from typing import Any, Dict, Optional

from fpgai.ir.graph import Graph


def _spec_to_c_type(spec: Dict[str, Any], default_total: int, default_int: int) -> str:
    if not isinstance(spec, dict):
        return f"ap_fixed<{default_total},{default_int}>"

    typ = str(spec.get("type", "ap_fixed")).lower()
    if typ in ("float", "float32"):
        return "float"
    if typ in ("double", "float64"):
        return "double"

    tb = int(spec.get("total_bits", default_total))
    ib = int(spec.get("int_bits", default_int))
    return f"ap_fixed<{tb},{ib}>"


def _default_precision() -> Dict[str, Dict[str, Any]]:
    return {
        "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    }


def _default_parallel() -> Dict[str, int]:
    return {
        "dense_in_unroll": 1,
        "conv_ic_unroll": 1,
    }


def emit_types_h(
    graph: Graph,
    *,
    top_name: str,
    numerics: Optional[Dict[str, Any]] = None,
    parallel: Optional[Dict[str, Any]] = None,
) -> str:
    dflt = _default_precision()
    global_prec = numerics if isinstance(numerics, dict) and numerics else dflt

    par = _default_parallel()
    if isinstance(parallel, dict):
        for k in list(par.keys()):
            if k in parallel:
                par[k] = max(1, int(parallel[k]))

    lines = []
    lines.append("#pragma once")
    lines.append("#include <ap_fixed.h>")
    lines.append("")
    lines.append(f"#define FPGAI_DENSE_IN_UNROLL {par['dense_in_unroll']}")
    lines.append(f"#define FPGAI_CONV_IC_UNROLL {par['conv_ic_unroll']}")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")

    # Global typedefs
    lines.append(f"typedef {_spec_to_c_type(global_prec.get('activation', dflt['activation']), 16, 6)} act_t;")
    lines.append(f"typedef {_spec_to_c_type(global_prec.get('weight', dflt['weight']), 16, 6)} wgt_t;")
    lines.append(f"typedef {_spec_to_c_type(global_prec.get('bias', dflt['bias']), 24, 10)} bias_t;")
    lines.append(f"typedef {_spec_to_c_type(global_prec.get('accum', dflt['accum']), 24, 10)} acc_t;")
    lines.append("typedef acc_t accum_t;")
    lines.append("")

    # Per-op typedefs required by params/codegen
    for idx, op in enumerate(graph.ops):
        p = op.attrs.get("precision", global_prec)
        tag = op.attrs.get("precision_tag", f"op{idx}")

        act_spec = p.get("activation", global_prec.get("activation", dflt["activation"]))
        wgt_spec = p.get("weight", global_prec.get("weight", dflt["weight"]))
        bias_spec = p.get("bias", global_prec.get("bias", dflt["bias"]))
        acc_spec = p.get("accum", global_prec.get("accum", dflt["accum"]))

        lines.append(f"// {idx}: {op.name} ({op.op_type})")
        lines.append(f"typedef {_spec_to_c_type(act_spec, 16, 6)} {tag}_act_t;")
        lines.append(f"typedef {_spec_to_c_type(wgt_spec, 16, 6)} {tag}_wgt_t;")
        lines.append(f"typedef {_spec_to_c_type(bias_spec, 24, 10)} {tag}_bias_t;")
        lines.append(f"typedef {_spec_to_c_type(acc_spec, 24, 10)} {tag}_acc_t;")
        lines.append("")

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)