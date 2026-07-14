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
    max_ref_abs = float(np.max(np.abs(ref))) if n else 0.0
    max_got_abs = float(np.max(np.abs(got))) if n else 0.0

    if ref_norm == 0.0 or got_norm == 0.0:
        cosine = 1.0 if np.allclose(ref, got) else 0.0
    else:
        cosine = float(np.dot(ref, got) / (ref_norm * got_norm))

    relative_l2 = float(l2 / ref_norm) if ref_norm > 0.0 else (0.0 if l2 == 0.0 else float("inf"))

    low_energy_ref = bool(ref_norm < 1e-5 or max_ref_abs < 1e-5)
    low_energy_got = bool(got_norm < 1e-5 or max_got_abs < 1e-5)
    cosine_reliable = bool(not low_energy_ref and not low_energy_got)

    return {
        "mae": mae,
        "max_abs": max_abs,
        "cosine": cosine,
        "l2": l2,
        "relative_l2": relative_l2,
        "ref_norm": ref_norm,
        "got_norm": got_norm,
        "max_ref_abs": max_ref_abs,
        "max_got_abs": max_got_abs,
        "low_energy_ref": low_energy_ref,
        "low_energy_got": low_energy_got,
        "cosine_reliable": cosine_reliable,
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
            "max_relative_l2": None,
            "min_ref_norm": None,
            "min_got_norm": None,
            "unreliable_cosine_count": 0,
        }

    cosines = np.asarray([float(x["cosine"]) for x in items], dtype=np.float32)
    maes = np.asarray([float(x["mae"]) for x in items], dtype=np.float32)
    max_abs = np.asarray([float(x["max_abs"]) for x in items], dtype=np.float32)
    rel_l2 = np.asarray([float(x["relative_l2"]) for x in items], dtype=np.float32)
    ref_norms = np.asarray([float(x["ref_norm"]) for x in items], dtype=np.float32)
    got_norms = np.asarray([float(x["got_norm"]) for x in items], dtype=np.float32)
    unreliable = [not bool(x["cosine_reliable"]) for x in items]

    return {
        "count": int(len(items)),
        "min_cosine": float(np.min(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "max_mae": float(np.max(maes)),
        "max_max_abs": float(np.max(max_abs)),
        "max_relative_l2": float(np.max(rel_l2)),
        "min_ref_norm": float(np.min(ref_norms)),
        "min_got_norm": float(np.min(got_norms)),
        "unreliable_cosine_count": int(sum(unreliable)),
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

    def _emit_global(title: str, m: Dict[str, Any]) -> None:
        lines.append(f"  {title}:")
        lines.append(f"    mae            : {m['mae']}")
        lines.append(f"    max_abs        : {m['max_abs']}")
        lines.append(f"    cosine         : {m['cosine']}")
        lines.append(f"    l2             : {m['l2']}")
        lines.append(f"    relative_l2    : {m['relative_l2']}")
        lines.append(f"    ref_norm       : {m['ref_norm']}")
        lines.append(f"    got_norm       : {m['got_norm']}")
        lines.append(f"    max_ref_abs    : {m['max_ref_abs']}")
        lines.append(f"    max_got_abs    : {m['max_got_abs']}")
        lines.append(f"    cosine_reliable: {m['cosine_reliable']}")

    _emit_global("grads", grads_m)
    _emit_global("weights_before", wb_m)
    _emit_global("weights_after", wa_m)
    _emit_global("weight_delta", wd_m)

    def _emit_group(title: str, key: str) -> None:
        items = grouped[key]
        summary = grouped_summary[key]
        lines.append(f"{title}:")
        lines.append(f"  count                 : {summary['count']}")
        lines.append(f"  min_cosine            : {summary['min_cosine']}")
        lines.append(f"  mean_cosine           : {summary['mean_cosine']}")
        lines.append(f"  max_mae               : {summary['max_mae']}")
        lines.append(f"  max_abs               : {summary['max_max_abs']}")
        lines.append(f"  max_relative_l2       : {summary['max_relative_l2']}")
        lines.append(f"  min_ref_norm          : {summary['min_ref_norm']}")
        lines.append(f"  min_got_norm          : {summary['min_got_norm']}")
        lines.append(f"  unreliable_cosine_cnt : {summary['unreliable_cosine_count']}")
        for item in items:
            lines.append(
                "  "
                f"{item['name']}: "
                f"cosine={item['cosine']:.6f} "
                f"mae={item['mae']:.6e} "
                f"max_abs={item['max_abs']:.6e} "
                f"rel_l2={item['relative_l2']:.6e} "
                f"ref_norm={item['ref_norm']:.6e} "
                f"got_norm={item['got_norm']:.6e} "
                f"cosine_reliable={item['cosine_reliable']}"
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

def build_dataset_training_comparison(
    *,
    training_compare_result: TrainingCompareResult | None,
    execution_payload: Dict[str, Any] | None,
    reference_payload: Dict[str, Any] | None,
    limits: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Build the canonical dataset-training comparison decision artifact.

    Raw comparison metrics are useful diagnostics, but their mere existence is
    not a validation result. This adapter verifies that the required vectors
    were compared completely, metrics are finite, configured tolerances pass,
    and HLS/reference execution counts agree.
    """
    thresholds = {
        "min_cosine_similarity": 0.99,
        "max_gradient_mae": 0.01,
        "max_gradient_max_abs": 0.05,
        "max_weight_delta_mae": 0.001,
        "max_weight_delta_max_abs": 0.01,
        "max_final_weight_mae": 0.001,
        "max_final_weight_max_abs": 0.01,
        "max_relative_l2": 0.10,
    }
    if limits:
        thresholds.update({str(k): float(v) for k, v in limits.items()})

    payload: Dict[str, Any] = {
        "artifact_kind": "fpgai_training_dataset_comparison",
        "schema_version": 1,
        "status": "pending_comparison",
        "passed": False,
        "limits": thresholds,
        "execution_comparison": None,
        "gradient_comparison": None,
        "weight_delta_comparison": None,
        "final_weight_comparison": None,
        "checks": [],
        "reason": "Required HLS/reference comparison artifacts are unavailable.",
    }
    if training_compare_result is None:
        return payload

    results_path = Path(training_compare_result.results_json)
    if not results_path.exists() or results_path.stat().st_size == 0:
        payload["reason"] = "Raw training comparison results are missing or empty."
        return payload
    try:
        raw = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload["reason"] = f"Raw training comparison results could not be read: {exc}"
        return payload
    if not isinstance(raw, dict) or not raw:
        payload["reason"] = "Raw training comparison results contain no metrics."
        return payload

    required = {"grads", "weight_delta", "weights_after"}
    missing = sorted(required.difference(raw))
    if missing:
        payload["reason"] = "Raw training comparison is missing sections: " + ", ".join(missing)
        return payload

    def finite(value: Any) -> bool:
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    checks: List[Dict[str, Any]] = []

    def check(name: str, value: Any, limit: Any, passed: bool, metric: str) -> None:
        checks.append({
            "name": name,
            "metric": metric,
            "value": value,
            "limit": limit,
            "passed": bool(passed),
        })

    def validate_vector(section_name: str, metric: Dict[str, Any], *, mae_limit: float, max_abs_limit: float) -> Dict[str, Any]:
        count = int(metric.get("count", 0) or 0)
        ref_count = int(metric.get("ref_count", 0) or 0)
        got_count = int(metric.get("got_count", 0) or 0)
        count_ok = count > 0 and count == ref_count == got_count
        check(f"{section_name}.vector_count", [ref_count, got_count], "equal_nonzero", count_ok, "count")

        mae = metric.get("mae")
        max_abs = metric.get("max_abs")
        relative_l2 = metric.get("relative_l2")
        cosine = metric.get("cosine")
        cosine_reliable = bool(metric.get("cosine_reliable", False))

        mae_ok = finite(mae) and float(mae) <= mae_limit
        max_abs_ok = finite(max_abs) and float(max_abs) <= max_abs_limit
        rel_ok = finite(relative_l2) and float(relative_l2) <= thresholds["max_relative_l2"]
        cosine_ok = (not cosine_reliable) or (
            finite(cosine) and float(cosine) >= thresholds["min_cosine_similarity"]
        )
        check(f"{section_name}.mae", mae, mae_limit, mae_ok, "mae")
        check(f"{section_name}.max_abs", max_abs, max_abs_limit, max_abs_ok, "max_abs")
        check(f"{section_name}.relative_l2", relative_l2, thresholds["max_relative_l2"], rel_ok, "relative_l2")
        check(
            f"{section_name}.cosine_similarity",
            cosine,
            thresholds["min_cosine_similarity"] if cosine_reliable else "not_required_low_energy",
            cosine_ok,
            "cosine_similarity",
        )
        return {
            **metric,
            "passed": bool(count_ok and mae_ok and max_abs_ok and rel_ok and cosine_ok),
        }

    payload["gradient_comparison"] = validate_vector(
        "gradients",
        dict(raw["grads"]),
        mae_limit=thresholds["max_gradient_mae"],
        max_abs_limit=thresholds["max_gradient_max_abs"],
    )
    payload["weight_delta_comparison"] = validate_vector(
        "weight_delta",
        dict(raw["weight_delta"]),
        mae_limit=thresholds["max_weight_delta_mae"],
        max_abs_limit=thresholds["max_weight_delta_max_abs"],
    )
    payload["final_weight_comparison"] = validate_vector(
        "final_weights",
        dict(raw["weights_after"]),
        mae_limit=thresholds["max_final_weight_mae"],
        max_abs_limit=thresholds["max_final_weight_max_abs"],
    )

    execution = execution_payload or {}
    reference = reference_payload or {}
    hls_samples = int(execution.get("sample_count_executed", execution.get("dataset_records_consumed", 0)) or 0)
    requested_samples = int(execution.get("sample_count_requested", 0) or 0)
    reference_samples = int(reference.get("sample_count", execution.get("reference_samples_executed", 0)) or 0)
    hls_updates = int(execution.get("optimizer_update_calls", 0) or 0)
    reference_updates = int(reference.get("optimizer_updates", 0) or 0)
    sample_ok = requested_samples > 0 and requested_samples == hls_samples == reference_samples
    update_ok = hls_updates > 0 and hls_updates == reference_updates
    check("execution.sample_count", [requested_samples, hls_samples, reference_samples], "all_equal_nonzero", sample_ok, "count")
    check("execution.optimizer_updates", [hls_updates, reference_updates], "equal_nonzero", update_ok, "count")
    payload["execution_comparison"] = {
        "sample_count_requested": requested_samples,
        "hls_samples_executed": hls_samples,
        "reference_samples_executed": reference_samples,
        "hls_optimizer_updates": hls_updates,
        "reference_optimizer_updates": reference_updates,
        "passed": bool(sample_ok and update_ok),
    }

    payload["checks"] = checks
    payload["raw_results_json"] = str(results_path)
    payload["raw_summary_txt"] = str(training_compare_result.summary_txt)
    passed = bool(checks) and all(bool(item["passed"]) for item in checks)
    payload["passed"] = passed
    payload["status"] = "passed" if passed else "failed_tolerance"
    failed = [item["name"] for item in checks if not item["passed"]]
    payload["reason"] = (
        "HLS and dataset-wide software-reference training artifacts satisfy all execution and numeric checks."
        if passed
        else "Training comparison failed check(s): " + ", ".join(failed)
    )
    return payload
