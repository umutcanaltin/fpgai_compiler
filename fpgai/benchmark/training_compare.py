from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from fpgai.util.fs import write_text


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


def _read_f32_bin(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError(f"Empty binary file: {path}")
    return arr


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise RuntimeError(f"Size mismatch: {a.size} vs {b.size}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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

    ref_g = _read_f32_bin(Path(ref_grads_bin))
    ref_wb = _read_f32_bin(Path(ref_weights_before_bin))
    ref_wa = _read_f32_bin(Path(ref_weights_after_bin))

    hls_g = _read_f32_bin(Path(hls_grads_bin))
    hls_wb = _read_f32_bin(Path(hls_weights_before_bin))
    hls_wa = _read_f32_bin(Path(hls_weights_after_bin))

    grad_cosine = _cosine(ref_g, hls_g)
    weight_after_cosine = _cosine(ref_wa, hls_wa)
    weight_delta_cosine = _cosine(ref_wa - ref_wb, hls_wa - hls_wb)

    grad_mae = float(np.mean(np.abs(ref_g - hls_g)))
    grad_max_abs = float(np.max(np.abs(ref_g - hls_g)))
    weight_after_mae = float(np.mean(np.abs(ref_wa - hls_wa)))
    weight_after_max_abs = float(np.max(np.abs(ref_wa - hls_wa)))

    payload = {
        "grad_cosine": grad_cosine,
        "weight_after_cosine": weight_after_cosine,
        "weight_delta_cosine": weight_delta_cosine,
        "grad_mae": grad_mae,
        "grad_max_abs": grad_max_abs,
        "weight_after_mae": weight_after_mae,
        "weight_after_max_abs": weight_after_max_abs,
        "ref": {
            "grads": str(ref_grads_bin),
            "weights_before": str(ref_weights_before_bin),
            "weights_after": str(ref_weights_after_bin),
        },
        "hls": {
            "grads": str(hls_grads_bin),
            "weights_before": str(hls_weights_before_bin),
            "weights_after": str(hls_weights_after_bin),
        },
    }

    results_json = out_dir / "results.json"
    summary_txt = out_dir / "summary.txt"
    write_text(results_json, json.dumps(payload, indent=2))
    write_text(
        summary_txt,
        "\n".join(
            [
                "=========== FPGAI Training Compare ===========",
                f"grad_cosine         : {grad_cosine}",
                f"weight_after_cosine : {weight_after_cosine}",
                f"weight_delta_cosine : {weight_delta_cosine}",
                f"grad_mae            : {grad_mae}",
                f"grad_max_abs        : {grad_max_abs}",
                f"weight_after_mae    : {weight_after_mae}",
                f"weight_after_max_abs: {weight_after_max_abs}",
                "==============================================",
            ]
        ),
    )

    return TrainingCompareResult(
        out_dir=out_dir,
        results_json=results_json,
        summary_txt=summary_txt,
        grad_cosine=grad_cosine,
        weight_after_cosine=weight_after_cosine,
        weight_delta_cosine=weight_delta_cosine,
        grad_mae=grad_mae,
        grad_max_abs=grad_max_abs,
        weight_after_mae=weight_after_mae,
        weight_after_max_abs=weight_after_max_abs,
    )