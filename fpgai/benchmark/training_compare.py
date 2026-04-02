from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np

from fpgai.util.fs import write_text


def _read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _safe_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.max(np.abs(a[:n] - b[:n])))


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    aa = a[:n].astype(np.float64, copy=False)
    bb = b[:n].astype(np.float64, copy=False)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def _safe_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    d = a[:n].astype(np.float64, copy=False) - b[:n].astype(np.float64, copy=False)
    return float(np.linalg.norm(d))


def _step_metrics(ref: np.ndarray, hw: np.ndarray) -> Dict[str, Any]:
    return {
        "length_ref": int(ref.size),
        "length_hw": int(hw.size),
        "compared_length": int(min(ref.size, hw.size)),
        "mae": _safe_mae(ref, hw),
        "max_abs": _safe_max_abs(ref, hw),
        "cosine": _safe_cosine(ref, hw),
        "l2": _safe_l2(ref, hw),
    }


def _maybe_compare_optional(ref_path: Optional[Path], hw_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if ref_path is None or hw_path is None:
        return None
    if (not ref_path.exists()) or (not hw_path.exists()):
        return None
    return _step_metrics(_read_f32(ref_path), _read_f32(hw_path))


@dataclass(frozen=True)
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


def compare_training_artifacts(
    *,
    out_dir: Path,
    ref_grads_bin: Path,
    ref_weights_before_bin: Path,
    ref_weights_after_bin: Path,
    hls_grads_bin: Path,
    hls_weights_before_bin: Path,
    hls_weights_after_bin: Path,
    ref_step_input_grad_bin: Optional[Path] = None,
    ref_step_output_bin: Optional[Path] = None,
    ref_step_target_bin: Optional[Path] = None,
    hls_step_input_grad_bin: Optional[Path] = None,
    hls_step_output_bin: Optional[Path] = None,
    hls_step_target_bin: Optional[Path] = None,
) -> TrainingCompareResult:
    cmp_dir = out_dir / "training_compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    ref_grads = _read_f32(ref_grads_bin)
    ref_w_before = _read_f32(ref_weights_before_bin)
    ref_w_after = _read_f32(ref_weights_after_bin)

    hls_grads = _read_f32(hls_grads_bin)
    hls_w_before = _read_f32(hls_weights_before_bin)
    hls_w_after = _read_f32(hls_weights_after_bin)

    ref_delta = ref_w_after - ref_w_before
    hls_delta = hls_w_after - hls_w_before

    grads_metrics = _step_metrics(ref_grads, hls_grads)
    weights_before_metrics = _step_metrics(ref_w_before, hls_w_before)
    weights_after_metrics = _step_metrics(ref_w_after, hls_w_after)
    weight_delta_metrics = _step_metrics(ref_delta, hls_delta)

    optional_steps = {
        "step_input_grad": _maybe_compare_optional(ref_step_input_grad_bin, hls_step_input_grad_bin),
        "step_output": _maybe_compare_optional(ref_step_output_bin, hls_step_output_bin),
        "step_target": _maybe_compare_optional(ref_step_target_bin, hls_step_target_bin),
    }

    payload = {
        "grads": grads_metrics,
        "weights_before": weights_before_metrics,
        "weights_after": weights_after_metrics,
        "weight_delta": weight_delta_metrics,
        "optional_steps": optional_steps,
        "paths": {
            "ref_grads_bin": str(ref_grads_bin),
            "ref_weights_before_bin": str(ref_weights_before_bin),
            "ref_weights_after_bin": str(ref_weights_after_bin),
            "hls_grads_bin": str(hls_grads_bin),
            "hls_weights_before_bin": str(hls_weights_before_bin),
            "hls_weights_after_bin": str(hls_weights_after_bin),
        },
    }

    results_json = cmp_dir / "results.json"
    summary_txt = cmp_dir / "summary.txt"

    write_text(results_json, json.dumps(payload, indent=2))

    lines = []
    lines.append("=============== FPGAI Training Compare ===============")
    lines.append("grads:")
    lines.append(f"  mae      : {grads_metrics['mae']}")
    lines.append(f"  max_abs  : {grads_metrics['max_abs']}")
    lines.append(f"  cosine   : {grads_metrics['cosine']}")
    lines.append(f"  l2       : {grads_metrics['l2']}")
    lines.append("weights_before:")
    lines.append(f"  mae      : {weights_before_metrics['mae']}")
    lines.append(f"  max_abs  : {weights_before_metrics['max_abs']}")
    lines.append(f"  cosine   : {weights_before_metrics['cosine']}")
    lines.append("weights_after:")
    lines.append(f"  mae      : {weights_after_metrics['mae']}")
    lines.append(f"  max_abs  : {weights_after_metrics['max_abs']}")
    lines.append(f"  cosine   : {weights_after_metrics['cosine']}")
    lines.append("weight_delta:")
    lines.append(f"  mae      : {weight_delta_metrics['mae']}")
    lines.append(f"  max_abs  : {weight_delta_metrics['max_abs']}")
    lines.append(f"  cosine   : {weight_delta_metrics['cosine']}")
    for k, v in optional_steps.items():
        if v is None:
            continue
        lines.append(f"{k}:")
        lines.append(f"  mae      : {v['mae']}")
        lines.append(f"  max_abs  : {v['max_abs']}")
        lines.append(f"  cosine   : {v['cosine']}")
        lines.append(f"  l2       : {v['l2']}")
    lines.append("======================================================")

    write_text(summary_txt, "\n".join(lines))

    return TrainingCompareResult(
        out_dir=cmp_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        grad_cosine=float(grads_metrics["cosine"]),
        weight_after_cosine=float(weights_after_metrics["cosine"]),
        weight_delta_cosine=float(weight_delta_metrics["cosine"]),
        grad_mae=float(grads_metrics["mae"]),
        grad_max_abs=float(grads_metrics["max_abs"]),
        weight_after_mae=float(weights_after_metrics["mae"]),
        weight_after_max_abs=float(weights_after_metrics["max_abs"]),
    )