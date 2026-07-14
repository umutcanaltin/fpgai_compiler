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
    float_training_compare_result: TrainingCompareResult | None = None,
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
        "reason": "Required HLS/hardware-domain reference comparison artifacts are unavailable.",
        "decision_reference_domain": "hardware_fixed_point",
        "float_reference_diagnostics": None,
    }
    if float_training_compare_result is not None:
        float_path = Path(float_training_compare_result.results_json)
        if float_path.exists() and float_path.stat().st_size > 0:
            try:
                payload["float_reference_diagnostics"] = json.loads(float_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload["float_reference_diagnostics"] = {"status": "unavailable"}
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
        "HLS and hardware-domain dataset training reference artifacts satisfy all execution and numeric checks."
        if passed
        else "Training comparison failed check(s): " + ", ".join(failed)
    )
    return payload


def build_training_semantic_trace_report(
    *,
    hls_gradient_accumulated: Path | None,
    hls_gradient_reduced: Path | None,
    ref_gradient_accumulated: Path | None,
    ref_gradient_reduced: Path | None,
    hls_weights_before: Path | None,
    hls_weights_after: Path | None,
) -> Dict[str, Any]:
    """Compare observable HLS training stages without inventing unavailable stages."""
    payload: Dict[str, Any] = {
        "artifact_kind": "fpgai_training_semantic_trace",
        "schema_version": 1,
        "status": "pending_trace",
        "first_divergence_stage": None,
        "stages": {},
        "availability": {},
        "reason": "Required semantic trace artifacts are unavailable.",
    }

    def compare_stage(name: str, ref_path: Path | None, hls_path: Path | None) -> None:
        available = bool(ref_path and hls_path and ref_path.exists() and hls_path.exists())
        payload["availability"][name] = {
            "reference": str(ref_path) if ref_path else None,
            "hls": str(hls_path) if hls_path else None,
            "available": available,
        }
        if not available:
            payload["stages"][name] = {"status": "not_available"}
            return
        ref = _read_f32(ref_path)  # type: ignore[arg-type]
        got = _read_f32(hls_path)  # type: ignore[arg-type]
        metrics = _metrics(ref, got)
        metrics["status"] = "compared"
        payload["stages"][name] = metrics

    compare_stage("gradient_accumulated_pre_reduce", ref_gradient_accumulated, hls_gradient_accumulated)
    compare_stage("gradient_reduced_export", ref_gradient_reduced, hls_gradient_reduced)

    weights_available = bool(
        hls_weights_before and hls_weights_after
        and hls_weights_before.exists() and hls_weights_after.exists()
    )
    payload["availability"]["weight_update_postcast"] = {
        "weights_before": str(hls_weights_before) if hls_weights_before else None,
        "weights_after": str(hls_weights_after) if hls_weights_after else None,
        "available": weights_available,
    }
    if weights_available:
        before = _read_f32(hls_weights_before)  # type: ignore[arg-type]
        after = _read_f32(hls_weights_after)  # type: ignore[arg-type]
        n = min(before.size, after.size)
        delta = after[:n] - before[:n]
        payload["stages"]["weight_update_postcast"] = {
            "status": "observed",
            "count": int(n),
            "l1_norm": float(np.sum(np.abs(delta))),
            "l2_norm": float(np.linalg.norm(delta)),
            "max_abs": float(np.max(np.abs(delta))) if n else 0.0,
        }
    else:
        payload["stages"]["weight_update_postcast"] = {"status": "not_available"}

    payload["stages"]["weight_update_precast"] = {
        "status": "not_observable",
        "reason": "The generated top currently exposes only post-cast parameter state.",
    }

    compared = [
        name for name in ("gradient_accumulated_pre_reduce", "gradient_reduced_export")
        if payload["stages"].get(name, {}).get("status") == "compared"
    ]
    if not compared:
        return payload

    payload["status"] = "available"
    payload["reason"] = "Observable HLS gradient stages were compared with the hardware-domain reference."
    # Diagnostic threshold only; this report identifies the first divergence and does not alter acceptance tolerances.
    for name in compared:
        stage = payload["stages"][name]
        if float(stage.get("relative_l2", 0.0)) > 0.10 or (
            bool(stage.get("cosine_reliable")) and float(stage.get("cosine", 1.0)) < 0.99
        ):
            payload["first_divergence_stage"] = name
            break
    if payload["first_divergence_stage"] is None:
        payload["first_divergence_stage"] = "after_observable_gradient_stages"
    return payload


def build_training_per_sample_gradient_trace_report(
    *,
    hls_trace_root: Path | None,
    ref_per_sample_paths: list[Path],
    ref_accumulator_paths: list[Path],
    parameter_layer_map_path: Path | None,
) -> Dict[str, Any]:
    """Compare per-sample gradients and accumulator snapshots stage by stage."""
    payload: Dict[str, Any] = {
        "artifact_kind": "fpgai_training_per_sample_gradient_trace",
        "schema_version": 1,
        "status": "pending_trace",
        "first_divergent_sample": None,
        "first_divergent_parameter_index": None,
        "first_divergent_layer": None,
        "first_divergent_role": None,
        "samples": [],
        "parameter_layer_map": None,
        "reason": "Per-sample trace artifacts are unavailable.",
    }
    layer_entries: list[dict[str, Any]] = []
    if parameter_layer_map_path and parameter_layer_map_path.exists():
        layer_payload = json.loads(parameter_layer_map_path.read_text(encoding="utf-8"))
        layer_entries = list(layer_payload.get("entries") or [])
        payload["parameter_layer_map"] = str(parameter_layer_map_path)

    def owner(index: int) -> tuple[str | None, str | None]:
        for entry in layer_entries:
            start = int(entry.get("offset", 0))
            count = int(entry.get("count", 0))
            if start <= index < start + count:
                return str(entry.get("layer")), str(entry.get("role"))
        return None, None

    if hls_trace_root is None or not hls_trace_root.exists() or not ref_per_sample_paths:
        return payload

    compared = 0
    for sample_index, ref_sample in enumerate(ref_per_sample_paths):
        hls_sample = hls_trace_root / f"per_sample_gradient_{sample_index:04d}.bin"
        hls_accum = hls_trace_root / f"accumulator_after_{sample_index:04d}.bin"
        ref_accum = ref_accumulator_paths[sample_index] if sample_index < len(ref_accumulator_paths) else None
        sample_payload: Dict[str, Any] = {"sample_index": sample_index}
        if not (ref_sample.exists() and hls_sample.exists()):
            sample_payload["gradient"] = {"status": "not_available"}
            payload["samples"].append(sample_payload)
            continue
        ref = _read_f32(ref_sample)
        got = _read_f32(hls_sample)
        metrics = _metrics(ref, got)
        metrics["status"] = "compared"
        sample_payload["gradient"] = metrics
        if ref_accum is not None and ref_accum.exists() and hls_accum.exists():
            sample_payload["accumulator"] = {"status": "compared", **_metrics(_read_f32(ref_accum), _read_f32(hls_accum))}
        else:
            sample_payload["accumulator"] = {"status": "not_available"}
        n = min(ref.size, got.size)
        if n:
            abs_err = np.abs(got[:n] - ref[:n])
            idx = int(np.argmax(abs_err))
            sample_payload["largest_error"] = {
                "parameter_index": idx,
                "reference_value": float(ref[idx]),
                "hls_value": float(got[idx]),
                "absolute_error": float(abs_err[idx]),
                "layer": owner(idx)[0],
                "role": owner(idx)[1],
            }
            divergent = float(metrics.get("relative_l2", 0.0)) > 0.10 or (
                bool(metrics.get("cosine_reliable")) and float(metrics.get("cosine", 1.0)) < 0.99
            )
            if divergent and payload["first_divergent_sample"] is None:
                layer, role = owner(idx)
                payload["first_divergent_sample"] = sample_index
                payload["first_divergent_parameter_index"] = idx
                payload["first_divergent_layer"] = layer
                payload["first_divergent_role"] = role
        payload["samples"].append(sample_payload)
        compared += 1

    if compared:
        payload["status"] = "available"
        payload["reason"] = "Per-sample gradients and accumulator snapshots were compared with the hardware-domain reference."
    return payload


def build_training_gradient_layer_role_reports(
    *,
    hls_trace_root: Path | None,
    ref_per_sample_paths: list[Path],
    ref_accumulator_paths: list[Path],
    parameter_layer_map_path: Path | None,
    batch_size: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build layer/role metrics and bias-specific scale/recurrence diagnostics."""
    base = {
        "schema_version": 1,
        "status": "pending_trace",
        "reason": "Per-sample trace artifacts or parameter map are unavailable.",
    }
    by_layer: Dict[str, Any] = {
        **base,
        "artifact_kind": "fpgai_training_gradient_by_layer",
        "layers": {},
    }
    by_role: Dict[str, Any] = {
        **base,
        "artifact_kind": "fpgai_training_gradient_by_role",
        "roles": {},
        "bias_trace": {"samples": []},
    }
    if hls_trace_root is None or not hls_trace_root.exists() or not ref_per_sample_paths:
        return by_layer, by_role
    if parameter_layer_map_path is None or not parameter_layer_map_path.exists():
        return by_layer, by_role

    entries = list(json.loads(parameter_layer_map_path.read_text(encoding="utf-8")).get("entries") or [])
    if not entries:
        return by_layer, by_role

    layer_acc: dict[str, dict[str, list[np.ndarray]]] = {}
    role_acc: dict[str, dict[str, list[np.ndarray]]] = {}
    bias_samples: list[dict[str, Any]] = []

    for sample_index, ref_path in enumerate(ref_per_sample_paths):
        hls_path = hls_trace_root / f"per_sample_gradient_{sample_index:04d}.bin"
        hls_acc_path = hls_trace_root / f"accumulator_after_{sample_index:04d}.bin"
        ref_acc_path = ref_accumulator_paths[sample_index] if sample_index < len(ref_accumulator_paths) else None
        if not (ref_path.exists() and hls_path.exists()):
            continue
        ref = _read_f32(ref_path)
        got = _read_f32(hls_path)
        sample_bias_ref: list[np.ndarray] = []
        sample_bias_got: list[np.ndarray] = []
        for entry in entries:
            start = int(entry.get("offset", 0))
            count = int(entry.get("count", 0))
            if count <= 0:
                continue
            stop = start + count
            layer = str(entry.get("layer", "unknown"))
            role = str(entry.get("role", "unknown"))
            r = ref[start:stop]
            g = got[start:stop]
            layer_acc.setdefault(layer, {}).setdefault(role, [[], []])[0].append(r)
            layer_acc[layer][role][1].append(g)
            role_acc.setdefault(role, {}).setdefault("all", [[], []])[0].append(r)
            role_acc[role]["all"][1].append(g)
            if role in {"bias", "beta"}:
                sample_bias_ref.append(r)
                sample_bias_got.append(g)
        if sample_bias_ref:
            r_bias = np.concatenate(sample_bias_ref)
            g_bias = np.concatenate(sample_bias_got)
            metrics = _metrics(r_bias, g_bias)
            ratio = float(metrics["got_norm"] / metrics["ref_norm"]) if metrics["ref_norm"] > 0 else None
            candidates = {
                "one": 1.0,
                "batch_size": float(batch_size),
                "inverse_batch_size": (1.0 / float(batch_size)) if batch_size else None,
            }
            finite_candidates = {k: v for k, v in candidates.items() if v is not None}
            best = min(finite_candidates, key=lambda k: abs((ratio if ratio is not None else 0.0) - finite_candidates[k])) if ratio is not None else None
            recurrence = None
            if ref_acc_path is not None and ref_acc_path.exists() and hls_acc_path.exists():
                ref_acc = _read_f32(ref_acc_path)
                hls_acc = _read_f32(hls_acc_path)
                bias_indices: list[int] = []
                for entry in entries:
                    if str(entry.get("role")) in {"bias", "beta"}:
                        start = int(entry.get("offset", 0)); count = int(entry.get("count", 0))
                        bias_indices.extend(range(start, start + count))
                idx = np.asarray(bias_indices, dtype=np.int64)
                recurrence = _metrics(ref_acc[idx], hls_acc[idx]) if idx.size else None
            bias_samples.append({
                "sample_index": sample_index,
                **metrics,
                "hls_to_reference_norm_ratio": ratio,
                "best_matching_scale": best,
                "scale_candidates": finite_candidates,
                "accumulator": recurrence,
            })

    def collapse(store: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for outer, inner in store.items():
            result[outer] = {}
            for role, pair in inner.items():
                refs, gots = pair
                if refs and gots:
                    result[outer][role] = _metrics(np.concatenate(refs), np.concatenate(gots))
        return result

    by_layer["layers"] = collapse(layer_acc)
    by_role["roles"] = collapse(role_acc)
    by_role["bias_trace"] = {
        "samples": bias_samples,
        "dominant_scale_counts": {
            name: sum(1 for item in bias_samples if item.get("best_matching_scale") == name)
            for name in ("one", "batch_size", "inverse_batch_size")
        },
    }
    by_layer["status"] = by_role["status"] = "available"
    by_layer["reason"] = "Per-sample gradients were aggregated by layer and parameter role."
    by_role["reason"] = "Per-sample gradients were aggregated by role with bias-specific scale diagnostics."
    return by_layer, by_role
