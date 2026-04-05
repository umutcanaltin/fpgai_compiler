from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
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
    }


def _find_layerwise_bins(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*.bin"):
        out[p.name] = p
    return out


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

    # layerwise compare
    ref_layer_root = Path(out_dir).parent / "training_reference" / "layerwise"
    hls_layer_root = hls_grads_bin.parent
    ref_bins = _find_layerwise_bins(ref_layer_root)
    hls_bins = _find_layerwise_bins(hls_layer_root)

    layerwise = []
    for name in sorted(set(ref_bins.keys()) & set(hls_bins.keys())):
        ref_arr = _read_f32(ref_bins[name])
        hls_arr = _read_f32(hls_bins[name])
        m = _metrics(ref_arr, hls_arr)
        m["name"] = name
        layerwise.append(m)

    results = {
        "grads": grads_m,
        "weights_before": wb_m,
        "weights_after": wa_m,
        "weight_delta": wd_m,
        "layerwise": layerwise,
        "ref_layerwise_dir": str(ref_layer_root),
        "hls_layerwise_dir": str(hls_layer_root),
    }

    results_json = out_dir / "results.json"
    summary_txt = out_dir / "summary.txt"
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("")
    lines.append("=============== FPGAI Training Compare ===============")
    lines.append("grads:")
    lines.append(f"  mae      : {grads_m['mae']}")
    lines.append(f"  max_abs  : {grads_m['max_abs']}")
    lines.append(f"  cosine   : {grads_m['cosine']}")
    lines.append(f"  l2       : {grads_m['l2']}")
    lines.append("weights_before:")
    lines.append(f"  mae      : {wb_m['mae']}")
    lines.append(f"  max_abs  : {wb_m['max_abs']}")
    lines.append(f"  cosine   : {wb_m['cosine']}")
    lines.append("weights_after:")
    lines.append(f"  mae      : {wa_m['mae']}")
    lines.append(f"  max_abs  : {wa_m['max_abs']}")
    lines.append(f"  cosine   : {wa_m['cosine']}")
    lines.append("weight_delta:")
    lines.append(f"  mae      : {wd_m['mae']}")
    lines.append(f"  max_abs  : {wd_m['max_abs']}")
    lines.append(f"  cosine   : {wd_m['cosine']}")
    if layerwise:
        lines.append("layerwise:")
        for item in layerwise:
            lines.append(
                f"  {item['name']}: cosine={item['cosine']:.6f} mae={item['mae']:.6e} max_abs={item['max_abs']:.6e}"
            )
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