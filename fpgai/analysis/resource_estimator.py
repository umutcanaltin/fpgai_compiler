from __future__ import annotations

from typing import Any, Dict, List


def _cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _default_spec(raw_cfg: Dict[str, Any], key: str, tb: int, ib: int) -> Dict[str, Any]:
    v = _cfg_get(raw_cfg, f"numerics.defaults.{key}", None)
    if isinstance(v, dict) and v.get("type") == "ap_fixed":
        return {
            "type": "ap_fixed",
            "total_bits": int(v.get("total_bits", tb)),
            "int_bits": int(v.get("int_bits", ib)),
        }
    return {"type": "ap_fixed", "total_bits": tb, "int_bits": ib}


def _match_rule(rule_match: Dict[str, Any], desc, idx: int) -> bool:
    if "name" in rule_match and str(rule_match["name"]) != str(desc.node_name):
        return False
    if "op_type" in rule_match and str(rule_match["op_type"]) != str(desc.op_type):
        return False
    if "index" in rule_match and int(rule_match["index"]) != idx:
        return False
    return True


def _precision_for_desc(raw_cfg: Dict[str, Any], desc, idx: int) -> Dict[str, Any]:
    p = {
        "activation": _default_spec(raw_cfg, "activation", 16, 6),
        "weight": _default_spec(raw_cfg, "weight", 16, 6),
        "bias": _default_spec(raw_cfg, "bias", 24, 10),
        "accum": _default_spec(raw_cfg, "accum", 24, 10),
    }
    rules = _cfg_get(raw_cfg, "numerics.layers", []) or []
    for rule in rules:
        match = rule.get("match", {})
        if not isinstance(match, dict):
            continue
        if not _match_rule(match, desc, idx):
            continue
        for k in ["activation", "weight", "bias", "accum"]:
            if k in rule and isinstance(rule[k], dict):
                p[k] = {
                    "type": "ap_fixed",
                    "total_bits": int(rule[k]["total_bits"]),
                    "int_bits": int(rule[k]["int_bits"]),
                }
    return p


def _estimate_layer_resources(desc, idx: int, raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = _precision_for_desc(raw_cfg, desc, idx)
    act_bits = int(p["activation"]["total_bits"])
    wgt_bits = int(p["weight"]["total_bits"])
    bias_bits = int(p["bias"]["total_bits"])
    acc_bits = int(p["accum"]["total_bits"])

    macs = int(getattr(desc, "macs", 0) or 0)
    param_bytes = int(getattr(desc, "param_bytes", 0) or 0)
    act_in = int(getattr(desc, "activation_bytes_in", 0) or 0)
    act_out = int(getattr(desc, "activation_bytes_out", 0) or 0)

    param_bits = param_bytes * 8
    act_bits_total = (act_in + act_out) * 8

    width_factor = (act_bits + wgt_bits + acc_bits) / 48.0
    storage_factor = (param_bits + act_bits_total) / 8.0

    predicted_lut = int(max(50, 100 + 0.0009 * macs * (act_bits + wgt_bits) + 0.002 * storage_factor))
    predicted_ff = int(max(80, 120 + 0.0012 * macs * (act_bits + acc_bits) + 0.003 * storage_factor))

    predicted_dsp = 0
    if desc.op_type in ("Conv", "Dense", "Gemm", "MatMul"):
        dsp_scale = max(0.25, (act_bits * wgt_bits) / 256.0)
        predicted_dsp = int(max(1, macs // 4096) * dsp_scale)

    predicted_bram18 = int(max(0, round((param_bits + act_bits_total) / 18432.0)))

    return {
        "layer_index": idx,
        "layer_name": desc.node_name,
        "op_type": desc.op_type,
        "act_bits": act_bits,
        "wgt_bits": wgt_bits,
        "bias_bits": bias_bits,
        "acc_bits": acc_bits,
        "macs": macs,
        "param_bytes": param_bytes,
        "activation_bytes_in": act_in,
        "activation_bytes_out": act_out,
        "predicted_lut": predicted_lut,
        "predicted_ff": predicted_ff,
        "predicted_dsp": predicted_dsp,
        "predicted_bram18": predicted_bram18,
    }


def estimate_resources_from_descriptors(descriptors: List[Any], raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    for idx, desc in enumerate(descriptors):
        layers.append(_estimate_layer_resources(desc, idx, raw_cfg))

    totals = {
        "predicted_lut": int(sum(x["predicted_lut"] for x in layers)),
        "predicted_ff": int(sum(x["predicted_ff"] for x in layers)),
        "predicted_dsp": int(sum(x["predicted_dsp"] for x in layers)),
        "predicted_bram18": int(sum(x["predicted_bram18"] for x in layers)),
        "total_macs": int(sum(x["macs"] for x in layers)),
    }

    return {
        "totals": totals,
        "worst_lut_layer": max(layers, key=lambda x: x["predicted_lut"]) if layers else None,
        "worst_dsp_layer": max(layers, key=lambda x: x["predicted_dsp"]) if layers else None,
        "worst_bram_layer": max(layers, key=lambda x: x["predicted_bram18"]) if layers else None,
        "layers": layers,
    }