from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import json
import numpy as np


@dataclass
class TrainingCompareResult:
    out_dir: Path
    results_json: Path
    summary_txt: Path
    grad_cosine: float
    weight_after_cosine: float
    weight_delta_cosine: float
    grad_mae: float
    grad_max_abs: float
    weight_after_mae: float
    weight_after_max_abs: float


def _read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _metrics(ref: np.ndarray, got: np.ndarray) -> Dict[str, float]:
    n = min(ref.size, got.size)
    ref = ref[:n].astype(np.float32)
    got = got[:n].astype(np.float32)

    diff = got - ref
    mae = float(np.mean(np.abs(diff))) if n else 0.0
    max_abs = float(np.max(np.abs(diff))) if n else 0.0
    l2 = float(np.sqrt(np.sum(diff * diff))) if n else 0.0

    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    if ref_norm == 0.0 or got_norm == 0.0:
        cosine = 1.0 if np.allclose(ref, got) else 0.0
    else:
        cosine = float(np.dot(ref, got) / (ref_norm * got_norm))

    return {
        "mae": mae,
        "max_abs": max_abs,
        "cosine": cosine,
        "l2": l2,
        "count": int(n),
        "ref_count": int(ref.size),
        "got_count": int(got.size),
    }


def _find_layerwise_bins(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*.bin"):
        out[p.name] = p
    return out


def _classify_layerwise_name(name: str) -> str:
    if name.endswith("__fwd.bin"):
        return "forward"
    if name.endswith("__bwd_in.bin"):
        return "backward_input"
    if name.endswith("__param_grad_w.bin"):
        return "param_grad_weight"
    if name.endswith("__param_grad_b.bin"):
        return "param_grad_bias"
    if name.endswith("__param_grad_gamma.bin"):
        return "param_grad_gamma"
    if name.endswith("__param_grad_beta.bin"):
        return "param_grad_beta"
    if name.endswith("__weights_before.bin"):
        return "weights_before"
    if name.endswith("__weights_after.bin"):
        return "weights_after"
    return "other"


def _group_layerwise_results(ref_bins: Dict[str, Path], hls_bins: Dict[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "forward_layerwise": [],
        "backward_input_layerwise": [],
        "param_grad_weight_layerwise": [],
        "param_grad_bias_layerwise": [],
        "param_grad_gamma_layerwise": [],
        "param_grad_beta_layerwise": [],
        "other_layerwise": [],
    }

    for name in sorted(set(ref_bins.keys()) & set(hls_bins.keys())):
        ref_arr = _read_f32(ref_bins[name])
        hls_arr = _read_f32(hls_bins[name])
        m = _metrics(ref_arr, hls_arr)
        item: Dict[str, Any] = {
            "name": name,
            "kind": _classify_layerwise_name(name),
            **m,
        }

        if item["kind"] == "forward":
            grouped["forward_layerwise"].append(item)
        elif item["kind"] == "backward_input":
            grouped["backward_input_layerwise"].append(item)
        elif item["kind"] == "param_grad_weight":
            grouped["param_grad_weight_layerwise"].append(item)
        elif item["kind"] == "param_grad_bias":
            grouped["param_grad_bias_layerwise"].append(item)
        elif item["kind"] == "param_grad_gamma":
            grouped["param_grad_gamma_layerwise"].append(item)
        elif item["kind"] == "param_grad_beta":
            grouped["param_grad_beta_layerwise"].append(item)
        else:
            grouped["other_layerwise"].append(item)

    return grouped


def _summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "min_cosine": None,
            "mean_cosine": None,
            "max_mae": None,
            "max_max_abs": None,
        }

    cosines = np.asarray([float(x["cosine"]) for x in items], dtype=np.float32)
    maes = np.asarray([float(x["mae"]) for x in items], dtype=np.float32)
    max_abs = np.asarray([float(x["max_abs"]) for x in items], dtype=np.float32)

    return {
        "count": int(len(items)),
        "min_cosine": float(np.min(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "max_mae": float(np.max(maes)),
        "max_max_abs": float(np.max(max_abs)),
    }


def compare_training_artifacts(
    *,
    out_dir: Path,
    ref_grads_bin: Path,
    ref_weights_before_bin: Path,
    ref_weights_after_bin: Path,
    hls_grads_bin: Path,
    hls_weights_before_bin: Path,
    hls_weights_after_bin: Path,
) -> TrainingCompareResult:
    out_dir = Path(out_dir) / "training_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_grads = _read_f32(ref_grads_bin)
    ref_w_before = _read_f32(ref_weights_before_bin)
    ref_w_after = _read_f32(ref_weights_after_bin)

    hls_grads = _read_f32(hls_grads_bin)
    hls_w_before = _read_f32(hls_weights_before_bin)
    hls_w_after = _read_f32(hls_weights_after_bin)

    ref_delta = ref_w_after - ref_w_before
    hls_delta = hls_w_after - hls_w_before

    grads_m = _metrics(ref_grads, hls_grads)
    wb_m = _metrics(ref_w_before, hls_w_before)
    wa_m = _metrics(ref_w_after, hls_w_after)
    wd_m = _metrics(ref_delta, hls_delta)

    ref_layer_root = Path(out_dir).parent / "training_reference" / "layerwise"
    hls_layer_root = hls_grads_bin.parent
    ref_bins = _find_layerwise_bins(ref_layer_root)
    hls_bins = _find_layerwise_bins(hls_layer_root)

    grouped = _group_layerwise_results(ref_bins, hls_bins)
    grouped_summary = {
        "forward_layerwise": _summarize_group(grouped["forward_layerwise"]),
        "backward_input_layerwise": _summarize_group(grouped["backward_input_layerwise"]),
        "param_grad_weight_layerwise": _summarize_group(grouped["param_grad_weight_layerwise"]),
        "param_grad_bias_layerwise": _summarize_group(grouped["param_grad_bias_layerwise"]),
        "param_grad_gamma_layerwise": _summarize_group(grouped["param_grad_gamma_layerwise"]),
        "param_grad_beta_layerwise": _summarize_group(grouped["param_grad_beta_layerwise"]),
        "other_layerwise": _summarize_group(grouped["other_layerwise"]),
    }

    results = {
        "grads": grads_m,
        "weights_before": wb_m,
        "weights_after": wa_m,
        "weight_delta": wd_m,
        **grouped,
        "layerwise_summary": grouped_summary,
        "ref_layerwise_dir": str(ref_layer_root),
        "hls_layerwise_dir": str(hls_layer_root),
    }

    results_json = out_dir / "results.json"
    summary_txt = out_dir / "summary.txt"
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("")
    lines.append("=============== FPGAI Training Compare ===============")
    lines.append("global:")
    lines.append("  grads:")
    lines.append(f"    mae      : {grads_m['mae']}")
    lines.append(f"    max_abs  : {grads_m['max_abs']}")
    lines.append(f"    cosine   : {grads_m['cosine']}")
    lines.append(f"    l2       : {grads_m['l2']}")
    lines.append("  weights_before:")
    lines.append(f"    mae      : {wb_m['mae']}")
    lines.append(f"    max_abs  : {wb_m['max_abs']}")
    lines.append(f"    cosine   : {wb_m['cosine']}")
    lines.append("  weights_after:")
    lines.append(f"    mae      : {wa_m['mae']}")
    lines.append(f"    max_abs  : {wa_m['max_abs']}")
    lines.append(f"    cosine   : {wa_m['cosine']}")
    lines.append("  weight_delta:")
    lines.append(f"    mae      : {wd_m['mae']}")
    lines.append(f"    max_abs  : {wd_m['max_abs']}")
    lines.append(f"    cosine   : {wd_m['cosine']}")

    def _emit_group(title: str, key: str) -> None:
        items = grouped[key]
        summary = grouped_summary[key]
        lines.append(f"{title}:")
        lines.append(f"  count      : {summary['count']}")
        lines.append(f"  min_cosine : {summary['min_cosine']}")
        lines.append(f"  mean_cosine: {summary['mean_cosine']}")
        lines.append(f"  max_mae    : {summary['max_mae']}")
        lines.append(f"  max_abs    : {summary['max_max_abs']}")
        for item in items:
            lines.append(
                f"  {item['name']}: cosine={item['cosine']:.6f} mae={item['mae']:.6e} max_abs={item['max_abs']:.6e}"
            )

    _emit_group("forward_layerwise", "forward_layerwise")
    _emit_group("backward_input_layerwise", "backward_input_layerwise")
    _emit_group("param_grad_weight_layerwise", "param_grad_weight_layerwise")
    _emit_group("param_grad_bias_layerwise", "param_grad_bias_layerwise")
    _emit_group("param_grad_gamma_layerwise", "param_grad_gamma_layerwise")
    _emit_group("param_grad_beta_layerwise", "param_grad_beta_layerwise")

    if grouped["other_layerwise"]:
        _emit_group("other_layerwise", "other_layerwise")

    lines.append("======================================================")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    return TrainingCompareResult(
        out_dir=out_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        grad_cosine=grads_m["cosine"],
        weight_after_cosine=wa_m["cosine"],
        weight_delta_cosine=wd_m["cosine"],
        grad_mae=grads_m["mae"],
        grad_max_abs=grads_m["max_abs"],
        weight_after_mae=wa_m["mae"],
        weight_after_max_abs=wa_m["max_abs"],
    )