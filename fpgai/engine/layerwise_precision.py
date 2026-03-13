from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def _cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _default_spec(raw: Dict[str, Any], key: str, tb: int, ib: int) -> Dict[str, Any]:
    v = _cfg_get(raw, f"numerics.defaults.{key}", None)
    if isinstance(v, dict) and v.get("type") == "ap_fixed":
        return {
            "type": "ap_fixed",
            "total_bits": int(v.get("total_bits", tb)),
            "int_bits": int(v.get("int_bits", ib)),
        }
    return {"type": "ap_fixed", "total_bits": tb, "int_bits": ib}


def _rule_matches(rule_match: Dict[str, Any], op, idx: int) -> bool:
    if "name" in rule_match and str(rule_match["name"]) != str(op.name):
        return False
    if "op_type" in rule_match and str(rule_match["op_type"]) != str(op.op_type):
        return False
    if "index" in rule_match and int(rule_match["index"]) != idx:
        return False
    return True


def resolve_layerwise_precision(graph, raw_cfg: Dict[str, Any]) -> None:
    defaults = {
        "activation": _default_spec(raw_cfg, "activation", 16, 6),
        "weight": _default_spec(raw_cfg, "weight", 16, 6),
        "bias": _default_spec(raw_cfg, "bias", 24, 10),
        "accum": _default_spec(raw_cfg, "accum", 24, 10),
    }

    rules = _cfg_get(raw_cfg, "numerics.layers", []) or []

    for idx, op in enumerate(graph.ops):
        p = deepcopy(defaults)

        for rule in rules:
            match = rule.get("match", {})
            if not isinstance(match, dict):
                continue
            if not _rule_matches(match, op, idx):
                continue

            for key in ["activation", "weight", "bias", "accum"]:
                if key in rule and isinstance(rule[key], dict):
                    p[key] = {
                        "type": "ap_fixed",
                        "total_bits": int(rule[key]["total_bits"]),
                        "int_bits": int(rule[key]["int_bits"]),
                    }

        op.attrs["precision"] = p
        op.attrs["precision_tag"] = f"op{idx}"