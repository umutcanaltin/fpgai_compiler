from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import json

import numpy as np


@dataclass(frozen=True)
class CompareMetrics:
    num_elements: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_similarity: float
    argmax_match: bool
    passed: bool

    atol: float
    rtol: float
    max_abs_error_limit: Optional[float]
    mean_abs_error_limit: Optional[float]
    rmse_limit: Optional[float]
    require_argmax_match: bool
    min_cosine_similarity: Optional[float]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)

    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def compare_outputs(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    max_abs_error_limit: float | None = None,
    mean_abs_error_limit: float | None = None,
    rmse_limit: float | None = None,
    require_argmax_match: bool = False,
    min_cosine_similarity: float | None = None,
) -> CompareMetrics:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)

    if ref.shape != cand.shape:
        raise ValueError(f"Shape mismatch: reference={ref.shape}, candidate={cand.shape}")

    diff = np.abs(ref - cand)
    max_abs_error = float(np.max(diff))
    mean_abs_error = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((ref - cand) ** 2)))
    cosine = _cosine_similarity(ref, cand)

    ref_argmax = int(np.argmax(ref.reshape(-1)))
    cand_argmax = int(np.argmax(cand.reshape(-1)))
    argmax_match = ref_argmax == cand_argmax

    checks = [bool(np.allclose(ref, cand, atol=atol, rtol=rtol))]

    if max_abs_error_limit is not None:
        checks.append(max_abs_error <= max_abs_error_limit)

    if mean_abs_error_limit is not None:
        checks.append(mean_abs_error <= mean_abs_error_limit)

    if rmse_limit is not None:
        checks.append(rmse <= rmse_limit)

    if min_cosine_similarity is not None:
        checks.append(cosine >= min_cosine_similarity)

    if require_argmax_match:
        checks.append(argmax_match)

    passed = all(checks)

    return CompareMetrics(
        num_elements=int(ref.size),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        rmse=rmse,
        cosine_similarity=cosine,
        argmax_match=argmax_match,
        passed=passed,
        atol=float(atol),
        rtol=float(rtol),
        max_abs_error_limit=max_abs_error_limit,
        mean_abs_error_limit=mean_abs_error_limit,
        rmse_limit=rmse_limit,
        require_argmax_match=bool(require_argmax_match),
        min_cosine_similarity=min_cosine_similarity,
    )


def write_metrics(metrics: CompareMetrics, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )

    lines = [
        "FPGAI Correctness Benchmark",
        "===========================",
        f"num_elements           : {metrics.num_elements}",
        f"max_abs_error          : {metrics.max_abs_error:.8f}",
        f"mean_abs_error         : {metrics.mean_abs_error:.8f}",
        f"rmse                   : {metrics.rmse:.8f}",
        f"cosine_similarity      : {metrics.cosine_similarity:.8f}",
        f"argmax_match           : {metrics.argmax_match}",
        f"passed                 : {metrics.passed}",
        f"atol                   : {metrics.atol}",
        f"rtol                   : {metrics.rtol}",
        f"max_abs_error_limit    : {metrics.max_abs_error_limit}",
        f"mean_abs_error_limit   : {metrics.mean_abs_error_limit}",
        f"rmse_limit             : {metrics.rmse_limit}",
        f"require_argmax_match   : {metrics.require_argmax_match}",
        f"min_cosine_similarity  : {metrics.min_cosine_similarity}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")