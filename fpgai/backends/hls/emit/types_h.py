from __future__ import annotations

from typing import Any, Dict
from fpgai.ir.graph import Graph


def _spec_to_ap(spec) -> str:
    if not isinstance(spec, dict):
        return "ap_fixed<16,6>"
    tb = int(spec.get("total_bits", 16))
    ib = int(spec.get("int_bits", 6))
    return f"ap_fixed<{tb},{ib}>"


def _deep_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _default_precision(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "activation": _deep_get(raw_cfg, "numerics.defaults.activation", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}),
        "weight": _deep_get(raw_cfg, "numerics.defaults.weight", {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}),
        "bias": _deep_get(raw_cfg, "numerics.defaults.bias", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}),
        "accum": _deep_get(raw_cfg, "numerics.defaults.accum", {"type": "ap_fixed", "total_bits": 24, "int_bits": 10}),
    }


def emit_types_h(graph: Graph, *, top_name: str, raw_cfg: Dict[str, Any] | None = None) -> str:
    raw_cfg = raw_cfg or {}
    dflt = _default_precision(raw_cfg)

    grad_act = _deep_get(raw_cfg, "numerics.training.grad_activation", dflt["activation"])
    grad_wgt = _deep_get(raw_cfg, "numerics.training.grad_weight", dflt["weight"])
    grad_bias = _deep_get(raw_cfg, "numerics.training.grad_bias", dflt["bias"])
    update_acc = _deep_get(raw_cfg, "numerics.training.update_accum", dflt["accum"])
    optimizer_state = _deep_get(raw_cfg, "numerics.training.optimizer_state", dflt["accum"])
    loss_t = _deep_get(raw_cfg, "numerics.training.loss", dflt["accum"])

    lines = []
    lines.append("#pragma once")
    lines.append("#include <ap_fixed.h>")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")
    lines.append(f"typedef {_spec_to_ap(dflt['activation'])} act_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['weight'])} wgt_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['bias'])} bias_t;")
    lines.append(f"typedef {_spec_to_ap(dflt['accum'])} acc_t;")
    lines.append("")
    lines.append(f"typedef {_spec_to_ap(grad_act)} grad_act_t;")
    lines.append(f"typedef {_spec_to_ap(grad_wgt)} grad_wgt_t;")
    lines.append(f"typedef {_spec_to_ap(grad_bias)} grad_bias_t;")
    lines.append(f"typedef {_spec_to_ap(update_acc)} upd_t;")
    lines.append(f"typedef {_spec_to_ap(optimizer_state)} opt_t;")
    lines.append(f"typedef {_spec_to_ap(loss_t)} loss_t;")
    lines.append("")

    for idx, op in enumerate(graph.ops):
        p = op.attrs.get("precision", dflt)
        tag = op.attrs.get("precision_tag", f"op{idx}")
        lines.append(f"typedef {_spec_to_ap(p['activation'])} {tag}_act_t;")
        lines.append(f"typedef {_spec_to_ap(p['weight'])} {tag}_wgt_t;")
        lines.append(f"typedef {_spec_to_ap(p['bias'])} {tag}_bias_t;")
        lines.append(f"typedef {_spec_to_ap(p['accum'])} {tag}_acc_t;")
        lines.append("")

    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)