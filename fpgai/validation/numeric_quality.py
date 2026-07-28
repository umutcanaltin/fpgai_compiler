"""Dataset execution and task-quality validation helpers."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .numeric_common import _cfg_lookup, _exists, _path_or_none, _read_f32_file

def _resolve_data_path(out_dir: str | Path, value: Any) -> Path | None:
    if value in (None, "", [], {}):
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    if path.is_absolute():
        return path
    candidates = [Path(out_dir) / path, Path.cwd() / path]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return candidates[0]


def _read_float_array(path: Any) -> list[float] | None:
    if path is None:
        return None
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        try:
            import numpy as np  # type: ignore
            return [float(x) for x in np.asarray(np.load(p), dtype=float).reshape(-1).tolist()]
        except Exception:
            return None
    if suffix in {".json", ".jsn"}:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            for key in ("values", "targets", "labels", "y", "data"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, list):
            try:
                return [float(x) for x in data]
            except Exception:
                return None
        return None
    if suffix in {".txt", ".csv", ".tsv"}:
        try:
            raw = p.read_text(encoding="utf-8").replace(",", " ").split()
            return [float(x) for x in raw]
        except Exception:
            return None
    return _read_f32_file(path)


def _read_int_array(path: Any) -> list[int] | None:
    floats = _read_float_array(path)
    if floats is not None:
        try:
            return [int(round(float(x))) for x in floats]
        except Exception:
            return None
    return None


def _reshape_samples(values: list[float], sample_count: int) -> list[list[float]] | None:
    if sample_count <= 0:
        return None
    if len(values) % sample_count != 0:
        return None
    width = len(values) // sample_count
    if width <= 0:
        return None
    return [values[i * width:(i + 1) * width] for i in range(sample_count)]


def _argmax(row: list[float]) -> int:
    if not row:
        return -1
    best = 0
    best_v = float(row[0])
    for idx, value in enumerate(row[1:], start=1):
        if float(value) > best_v:
            best = idx
            best_v = float(value)
    return best


def _topk(row: list[float], k: int) -> set[int]:
    return {idx for idx, _ in sorted(enumerate(row), key=lambda item: float(item[1]), reverse=True)[:max(1, k)]}


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _r2_score(targets: list[float], preds: list[float]) -> float | None:
    n = min(len(targets), len(preds))
    if n == 0:
        return None
    y = [float(targets[i]) for i in range(n)]
    p = [float(preds[i]) for i in range(n)]
    mean_y = sum(y) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    if ss_tot == 0.0:
        return None
    ss_res = sum((p[i] - y[i]) ** 2 for i in range(n))
    return 1.0 - ss_res / ss_tot


def _regression_metrics(targets: list[float], preds: list[float]) -> dict[str, Any]:
    n = min(len(targets), len(preds))
    if n == 0:
        return {"sample_count": 0, "mae": None, "rmse": None, "max_abs_error": None, "r2": None}
    diffs = [float(preds[i]) - float(targets[i]) for i in range(n)]
    abs_diffs = [abs(x) for x in diffs]
    mse = sum(x * x for x in diffs) / n
    return {
        "sample_count": n,
        "mae": sum(abs_diffs) / n,
        "rmse": math.sqrt(mse),
        "max_abs_error": max(abs_diffs),
        "r2": _r2_score(targets[:n], preds[:n]),
    }


def _classification_decision(payload: dict[str, Any], thresholds: Mapping[str, Any]) -> tuple[str, str]:
    accuracy_drop = payload.get("accuracy_drop_pct")
    agreement = payload.get("prediction_agreement_vs_reference")
    max_drop = float(thresholds.get("max_accuracy_drop_pct", 3.0) or 3.0)
    low_drop = float(thresholds.get("recommended_accuracy_drop_pct", 1.0) or 1.0)
    aggressive_drop = float(thresholds.get("aggressive_accuracy_drop_pct", 10.0) or 10.0)
    min_agree = float(thresholds.get("min_prediction_agreement", 0.95) or 0.95)
    aggressive_agree = float(thresholds.get("aggressive_min_prediction_agreement", 0.75) or 0.75)

    if accuracy_drop is not None:
        drop = abs(float(accuracy_drop))
        if drop <= low_drop and (agreement is None or float(agreement) >= 0.99):
            return "recommended_quality", f"accuracy drop {drop:.3g}% is within the recommended threshold"
        if drop <= max_drop and (agreement is None or float(agreement) >= min_agree):
            return "acceptable_tradeoff", f"accuracy drop {drop:.3g}% is within the configured decision threshold"
        if drop <= aggressive_drop and (agreement is None or float(agreement) >= aggressive_agree):
            return "aggressive_compression", f"accuracy drop {drop:.3g}% may be acceptable for resource-constrained deployments"
        return "not_recommended_for_quality", f"accuracy drop {drop:.3g}% exceeds the configured decision threshold"

    if agreement is not None:
        agree = float(agreement)
        if agree >= 0.99:
            return "recommended_quality", f"prediction agreement {agree:.3g} is near-identical to the reference"
        if agree >= min_agree:
            return "acceptable_tradeoff", f"prediction agreement {agree:.3g} is within the configured threshold"
        if agree >= aggressive_agree:
            return "aggressive_compression", f"prediction agreement {agree:.3g} may be acceptable for aggressive compression"
        return "not_recommended_for_quality", f"prediction agreement {agree:.3g} is below the configured threshold"
    return "pending_dataset_labels", "classification labels were not provided and reference prediction agreement could not be computed"


def _regression_decision(payload: dict[str, Any], thresholds: Mapping[str, Any]) -> tuple[str, str]:
    mae_inc = payload.get("mae_increase")
    rmse_inc = payload.get("rmse_increase")
    ref_mae = payload.get("reference_output_mae")
    max_mae_inc = float(thresholds.get("max_mae_increase", 0.01) or 0.01)
    max_rmse_inc = float(thresholds.get("max_rmse_increase", 0.02) or 0.02)
    if mae_inc is not None or rmse_inc is not None:
        mi = abs(float(mae_inc or 0.0))
        ri = abs(float(rmse_inc or 0.0))
        if mi <= max_mae_inc * 0.25 and ri <= max_rmse_inc * 0.25:
            return "recommended_quality", "regression error increase is very small versus the reference"
        if mi <= max_mae_inc and ri <= max_rmse_inc:
            return "acceptable_tradeoff", "regression error increase is within the configured decision threshold"
        if mi <= max_mae_inc * 4.0 and ri <= max_rmse_inc * 4.0:
            return "aggressive_compression", "regression error increase is above the preferred threshold but may be acceptable for resource-constrained deployments"
        return "not_recommended_for_quality", "regression error increase exceeds the configured decision threshold"
    if ref_mae is not None:
        mae = abs(float(ref_mae))
        if mae <= max_mae_inc * 0.25:
            return "recommended_quality", "output MAE versus reference is very small"
        if mae <= max_mae_inc:
            return "acceptable_tradeoff", "output MAE versus reference is within the configured threshold"
        if mae <= max_mae_inc * 4.0:
            return "aggressive_compression", "output MAE versus reference is above the preferred threshold but may be acceptable"
        return "not_recommended_for_quality", "output MAE versus reference exceeds the configured threshold"
    return "pending_targets", "regression targets were not provided and output-error metrics were unavailable"


def _dataset_execution_validation(out_dir: str | Path, *, expected_sample_count: int | None = None, output_values_per_sample: int | None = None) -> dict[str, Any]:
    path = Path(out_dir) / "reports" / "hls_dataset_execution.json"
    base = {
        "artifact_path": str(path),
        "artifact_exists": path.exists(),
        "status": "not_requested",
        "passed": False,
        "checks": [],
    }
    if expected_sample_count is None or expected_sample_count <= 1:
        return base
    if not path.exists():
        # Compatibility path for reference-only unit tests and non-CSim callers.
        # When CSim emits a record, it is validated strictly below.
        return base
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        base.update({"status": "unreadable", "reason": str(exc)})
        return base
    requested = int(record.get("sample_count_requested") or 0)
    executed = int(record.get("sample_count_executed") or 0)
    invocations = int(record.get("inference_invocation_count") or 0)
    generated_words = int(record.get("generated_output_words") or 0)
    per_sample = int(record.get("output_values_per_sample") or output_values_per_sample or 0)
    expected_words = int(expected_sample_count) * int(per_sample)
    checks = [
        {"name": "requested_sample_count", "expected": expected_sample_count, "actual": requested, "passed": requested == expected_sample_count},
        {"name": "executed_sample_count", "expected": expected_sample_count, "actual": executed, "passed": executed == expected_sample_count},
        {"name": "inference_invocation_count", "expected": expected_sample_count, "actual": invocations, "passed": invocations == expected_sample_count},
        {"name": "generated_output_words", "expected": expected_words, "actual": generated_words, "passed": expected_words > 0 and generated_words == expected_words},
    ]
    passed = all(bool(item["passed"]) for item in checks)
    base.update({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "record": record,
    })
    return base


def _classification_diagnostics(labels: list[int], predictions: list[int], class_count: int) -> dict[str, Any]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    support = [0 for _ in range(class_count)]
    correct = [0 for _ in range(class_count)]
    for target, predicted in zip(labels, predictions):
        if 0 <= target < class_count:
            support[target] += 1
            if 0 <= predicted < class_count:
                matrix[target][predicted] += 1
            if predicted == target:
                correct[target] += 1
    per_class = []
    for class_id in range(class_count):
        per_class.append({
            "class_id": class_id,
            "support": support[class_id],
            "correct": correct[class_id],
            "accuracy": (correct[class_id] / support[class_id]) if support[class_id] else None,
        })
    return {"confusion_matrix": matrix, "per_class_accuracy": per_class}


def _task_quality_payload(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    output_compare: dict[str, Any] | None,
    inference_reference_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = raw_config or {}
    validation = _cfg_lookup(raw, "validation", {})
    if not isinstance(validation, dict):
        validation = {}
    task = str(validation.get("task") or validation.get("type") or "auto").strip().lower().replace("-", "_")
    thresholds = validation.get("decision_thresholds", {}) if isinstance(validation.get("decision_thresholds", {}), dict) else {}
    ref_path = (inference_reference_artifacts or {}).get("outputs_ref")
    hw_path = (inference_reference_artifacts or {}).get("outputs_hw")
    ref_values = _read_float_array(ref_path)
    hw_values = _read_float_array(hw_path)
    base: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_run",
        "task": task,
        "decision_status": "pending_numeric_artifacts",
        "decision_reason": "reference and generated output artifacts were not available",
        "reference_outputs_path": _path_or_none(ref_path),
        "generated_outputs_path": _path_or_none(hw_path),
        "reference_outputs_exist": _exists(ref_path),
        "generated_outputs_exist": _exists(hw_path),
        "dataset_source": ((validation.get("dataset") or {}).get("source") if isinstance(validation.get("dataset"), dict) else validation.get("dataset")) or validation.get("dataset_name") or validation.get("source") or "not_provided",
    }
    if ref_values is None or hw_values is None:
        return base
    if task == "auto":
        # Without labels or targets, default to output-agreement reporting. If labels
        # exist, classification takes priority; if targets exist, regression does.
        if validation.get("labels") or validation.get("label_path"):
            task = "classification"
        elif validation.get("targets") or validation.get("target_path"):
            task = "regression"
        else:
            task = "classification" if len(ref_values) > 1 else "regression"
    base["task"] = task

    if task in {"classification", "multiclass", "categorical"}:
        labels_path = _resolve_data_path(out_dir, validation.get("labels") or validation.get("label_path") or validation.get("y") or (inference_reference_artifacts or {}).get("labels_path"))
        labels = _read_int_array(labels_path) if labels_path is not None else None
        if labels:
            sample_count = len(labels)
            ref_rows = _reshape_samples(ref_values, sample_count)
            hw_rows = _reshape_samples(hw_values, sample_count)
        else:
            sample_count = 1
            ref_rows = [ref_values]
            hw_rows = [hw_values]
        if ref_rows is None or hw_rows is None or len(ref_rows) != len(hw_rows):
            base.update({
                "status": "shape_mismatch",
                "decision_status": "not_recommended_for_quality",
                "decision_reason": "classification output size could not be aligned with labels/samples",
                "labels_path": _path_or_none(labels_path),
                "labels_status": "shape_mismatch" if labels_path is not None else "not_provided",
            })
            return base
        ref_top1 = [_argmax(row) for row in ref_rows]
        hw_top1 = [_argmax(row) for row in hw_rows]
        agreement_count = sum(1 for a, b in zip(ref_top1, hw_top1) if a == b)
        agreement = agreement_count / len(ref_top1) if ref_top1 else None
        class_changes = len(ref_top1) - agreement_count
        conf_deltas = [float(max(hw_rows[i]) - max(ref_rows[i])) for i in range(len(ref_rows)) if ref_rows[i] and hw_rows[i]]
        execution_validation = _dataset_execution_validation(
            out_dir,
            expected_sample_count=len(ref_top1),
            output_values_per_sample=(len(ref_rows[0]) if ref_rows else None),
        )
        payload: dict[str, Any] = {
            **base,
            "status": "compared",
            "execution_validation": execution_validation,
            "labels_path": _path_or_none(labels_path),
            "labels_status": "provided" if labels else "not_provided",
            "sample_count": len(ref_top1),
            "class_count": len(ref_rows[0]) if ref_rows else 0,
            "reference_top1": ref_top1[:16],
            "generated_top1": hw_top1[:16],
            "reference_top1_first": ref_top1[0] if ref_top1 else None,
            "generated_top1_first": hw_top1[0] if hw_top1 else None,
            "prediction_agreement_vs_reference": agreement,
            "class_change_count": class_changes,
            "confidence_delta_mean": _safe_mean(conf_deltas),
            "confidence_delta_max": max([abs(x) for x in conf_deltas]) if conf_deltas else None,
            "target_top1_accuracy": None,
            "generated_top1_accuracy": None,
            "reference_top1_accuracy": None,
            "top1_accuracy_drop_pct": None,
            "generated_top5_accuracy": None,
            "reference_top5_accuracy": None,
            "top5_accuracy_drop_pct": None,
        }
        if labels:
            ref_correct = sum(1 for pred, y in zip(ref_top1, labels) if pred == y)
            hw_correct = sum(1 for pred, y in zip(hw_top1, labels) if pred == y)
            ref_acc = ref_correct / len(labels)
            hw_acc = hw_correct / len(labels)
            k = min(5, len(ref_rows[0]) if ref_rows else 1)
            ref_top5_correct = sum(1 for row, y in zip(ref_rows, labels) if y in _topk(row, k))
            hw_top5_correct = sum(1 for row, y in zip(hw_rows, labels) if y in _topk(row, k))
            ref_diagnostics = _classification_diagnostics(labels, ref_top1, len(ref_rows[0]) if ref_rows else 0)
            hw_diagnostics = _classification_diagnostics(labels, hw_top1, len(hw_rows[0]) if hw_rows else 0)
            diagnostics_dir = Path(out_dir) / "reports"
            confusion_path = diagnostics_dir / "classification_confusion_matrix.json"
            per_class_path = diagnostics_dir / "classification_per_class_accuracy.json"
            confusion_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "classification_confusion_matrix",
                "labels": list(range(len(ref_rows[0]) if ref_rows else 0)),
                "reference": ref_diagnostics["confusion_matrix"],
                "generated": hw_diagnostics["confusion_matrix"],
            }, indent=2), encoding="utf-8")
            per_class_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "classification_per_class_accuracy",
                "reference": ref_diagnostics["per_class_accuracy"],
                "generated": hw_diagnostics["per_class_accuracy"],
            }, indent=2), encoding="utf-8")
            payload.update({
                "target_top1_accuracy": hw_acc,
                "generated_top1_accuracy": hw_acc,
                "reference_top1_accuracy": ref_acc,
                "top1_accuracy_drop_pct": (hw_acc - ref_acc) * 100.0,
                "generated_top5_accuracy": hw_top5_correct / len(labels),
                "reference_top5_accuracy": ref_top5_correct / len(labels),
                "top5_accuracy_drop_pct": ((hw_top5_correct - ref_top5_correct) / len(labels)) * 100.0,
                "confusion_matrix_path": str(confusion_path),
                "per_class_accuracy_path": str(per_class_path),
                "reference_per_class_accuracy": ref_diagnostics["per_class_accuracy"],
                "generated_per_class_accuracy": hw_diagnostics["per_class_accuracy"],
            })
        if execution_validation.get("status") not in {"not_requested", "passed"}:
            payload["status"] = "execution_record_invalid"
            payload["decision_status"] = "not_recommended_for_quality"
            payload["decision_reason"] = "dataset HLS execution record is missing or inconsistent with the evaluated sample count"
            return payload
        decision, reason = _classification_decision({
            "accuracy_drop_pct": payload.get("top1_accuracy_drop_pct"),
            "prediction_agreement_vs_reference": agreement,
        }, thresholds)
        payload["decision_status"] = decision
        payload["decision_reason"] = reason
        return payload

    if task in {"regression", "numeric_regression"}:
        targets_path = _resolve_data_path(out_dir, validation.get("targets") or validation.get("target_path") or validation.get("y") or (inference_reference_artifacts or {}).get("targets_path"))
        targets = _read_float_array(targets_path) if targets_path is not None else None
        payload = {
            **base,
            "status": "compared",
            "targets_path": _path_or_none(targets_path),
            "targets_status": "provided" if targets is not None else "not_provided",
            "sample_count": min(len(ref_values), len(hw_values)),
            "reference_output_mae": output_compare.get("mae") if output_compare else None,
            "reference_output_rmse": math.sqrt(float(output_compare.get("mse"))) if output_compare and output_compare.get("mse") is not None and float(output_compare.get("mse")) >= 0 else None,
            "reference_output_max_abs_error": output_compare.get("max_abs_error") if output_compare else None,
            "target_mae_reference": None,
            "target_mae_generated": None,
            "target_rmse_reference": None,
            "target_rmse_generated": None,
            "target_r2_reference": None,
            "target_r2_generated": None,
            "mae_increase": None,
            "rmse_increase": None,
        }
        if targets is not None:
            ref_m = _regression_metrics(targets, ref_values)
            hw_m = _regression_metrics(targets, hw_values)
            payload.update({
                "sample_count": min(ref_m.get("sample_count") or 0, hw_m.get("sample_count") or 0),
                "target_mae_reference": ref_m.get("mae"),
                "target_mae_generated": hw_m.get("mae"),
                "target_rmse_reference": ref_m.get("rmse"),
                "target_rmse_generated": hw_m.get("rmse"),
                "target_r2_reference": ref_m.get("r2"),
                "target_r2_generated": hw_m.get("r2"),
                "mae_increase": None if ref_m.get("mae") is None or hw_m.get("mae") is None else float(hw_m["mae"]) - float(ref_m["mae"]),
                "rmse_increase": None if ref_m.get("rmse") is None or hw_m.get("rmse") is None else float(hw_m["rmse"]) - float(ref_m["rmse"]),
            })
        decision, reason = _regression_decision(payload, thresholds)
        payload["decision_status"] = decision
        payload["decision_reason"] = reason
        return payload

    base.update({
        "status": "unsupported_task",
        "decision_status": "pending_task_metadata",
        "decision_reason": f"validation.task={task!r} is not supported by task-quality reporting",
    })
    return base
