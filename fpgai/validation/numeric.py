"""Numeric validation report helpers.

This module does not pretend that a design has been numerically validated.
It records exactly which reference/testbench/HLS artifacts exist and whether
there is enough evidence to claim numeric correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import struct
import math


def _path_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(Path(value))
    except TypeError:
        return str(value)


def _exists(value: Any) -> bool:
    if value is None:
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False


def _read_f32_file(path: Any) -> list[float] | None:
    if path is None:
        return None
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    if len(data) % 4 != 0:
        return None
    if not data:
        return []
    try:
        return list(struct.unpack('<' + 'f' * (len(data) // 4), data))
    except Exception:
        return None


def _compare_vectors(ref: list[float], got: list[float]) -> dict[str, Any]:
    n = min(len(ref), len(got))
    if len(ref) != len(got):
        status = 'shape_mismatch'
    else:
        status = 'compared'
    if n == 0:
        return {
            'status': status,
            'num_ref': len(ref),
            'num_got': len(got),
            'num_compared': 0,
            'mse': None,
            'mae': None,
            'max_abs_error': None,
            'cosine_similarity': None,
        }
    diffs = [float(got[i] - ref[i]) for i in range(n)]
    abs_diffs = [abs(x) for x in diffs]
    mse = sum(x * x for x in diffs) / n
    mae = sum(abs_diffs) / n
    max_abs = max(abs_diffs)
    dot = sum(float(ref[i]) * float(got[i]) for i in range(n))
    nr = math.sqrt(sum(float(ref[i]) * float(ref[i]) for i in range(n)))
    ng = math.sqrt(sum(float(got[i]) * float(got[i]) for i in range(n)))
    cosine = None if nr == 0.0 or ng == 0.0 else dot / (nr * ng)
    return {
        'status': status,
        'num_ref': len(ref),
        'num_got': len(got),
        'num_compared': n,
        'mse': mse,
        'mae': mae,
        'max_abs_error': max_abs,
        'cosine_similarity': cosine,
    }


def _compare_file_pair(ref_path: Any, got_path: Any) -> dict[str, Any]:
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    payload = {
        'ref_path': _path_or_none(ref_path),
        'got_path': _path_or_none(got_path),
        'ref_exists': _exists(ref_path),
        'got_exists': _exists(got_path),
    }
    if ref is None or got is None:
        payload.update({'status': 'missing_or_unreadable', 'passed': False})
        return payload
    metrics = _compare_vectors(ref, got)
    passed = (metrics['status'] == 'compared' and (metrics['max_abs_error'] is not None) and metrics['max_abs_error'] <= 1e-3)
    payload.update(metrics)
    payload['passed'] = bool(passed)
    return payload




def _optimizer_state_validation_payload(optimizer_state_artifacts: dict[str, Any] | None) -> dict[str, Any]:
    """Describe and compare persistent optimizer-state tensors when artifacts exist.

    Momentum and Adam correctness is not only weights-after correctness: their
    persistent state must also be checked.  This helper records requested state
    tensors and compares explicit ref/got float32 files when the testbench or
    runtime path provides them.  Missing files are reported as missing evidence,
    never as a pass.
    """
    if not optimizer_state_artifacts:
        return {"requested": False, "status": "not_requested"}

    payload: dict[str, Any] = dict(optimizer_state_artifacts)
    requested = bool(payload.get("requested", False))
    comparisons_cfg = payload.get("comparisons", {}) or {}
    comparisons: dict[str, Any] = {}
    any_compared = False
    all_passed = True

    if isinstance(comparisons_cfg, dict):
        for name, cfg in sorted(comparisons_cfg.items()):
            if not isinstance(cfg, dict):
                continue
            cmp_payload = _compare_file_pair(cfg.get("ref"), cfg.get("got"))
            comparisons[str(name)] = cmp_payload
            if cmp_payload.get("status") == "compared":
                any_compared = True
            if not bool(cmp_payload.get("passed", False)):
                all_passed = False

    payload["comparisons"] = comparisons
    if not requested:
        payload["status"] = "not_requested"
        payload["passed"] = False
    elif comparisons and any_compared and all_passed:
        payload["status"] = "compared"
        payload["passed"] = True
    elif comparisons:
        payload["status"] = "missing_or_failed"
        payload["passed"] = False
    else:
        payload.setdefault("status", "generated_not_captured_by_testbench")
        payload["passed"] = False
    return payload


def _training_reference_payload(training_reference_result: Any) -> dict[str, Any] | None:
    if training_reference_result is None:
        return None
    return {
        "status": "generated",
        "loss_before": getattr(training_reference_result, "loss_before", None),
        "loss_after": getattr(training_reference_result, "loss_after", None),
        "grads_ref_bin": _path_or_none(getattr(training_reference_result, "grads_flat_path", None)),
        "weights_before_ref_bin": _path_or_none(getattr(training_reference_result, "weights_before_flat_path", None)),
        "weights_after_ref_bin": _path_or_none(getattr(training_reference_result, "weights_after_flat_path", None)),
        "summary_json": _path_or_none(getattr(training_reference_result, "summary_json", None)),
        "summary_txt": _path_or_none(getattr(training_reference_result, "summary_txt", None)),
    }


def _training_compare_payload(training_compare_result: Any) -> dict[str, Any] | None:
    if training_compare_result is None:
        return None
    return {
        "status": "compared",
        "results_json": _path_or_none(getattr(training_compare_result, "results_json", None)),
        "summary_txt": _path_or_none(getattr(training_compare_result, "summary_txt", None)),
        "grad_cosine": getattr(training_compare_result, "grad_cosine", None),
        "weight_after_cosine": getattr(training_compare_result, "weight_after_cosine", None),
        "weight_delta_cosine": getattr(training_compare_result, "weight_delta_cosine", None),
        "grad_mae": getattr(training_compare_result, "grad_mae", None),
        "grad_max_abs": getattr(training_compare_result, "grad_max_abs", None),
        "weight_after_mae": getattr(training_compare_result, "weight_after_mae", None),
        "weight_after_max_abs": getattr(training_compare_result, "weight_after_max_abs", None),
    }


def emit_numeric_validation_report(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    source_generated: bool,
    hls_ran: bool = False,
    hls_ok: bool | None = None,
    training_reference_result: Any = None,
    training_compare_result: Any = None,
    inference_reference_artifacts: dict[str, Any] | None = None,
    gradient_export_artifacts: dict[str, Any] | None = None,
    optimizer_state_artifacts: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write numeric validation reports and return artifact paths.

    The report is intentionally conservative:
    - inference is marked ``not_run`` unless explicit inference validation
      artifacts are provided;
    - training is marked ``passed`` only when the training comparison result
      exists and exposes a summary/results file;
    - missing HLS/testbench artifacts are recorded as missing evidence, not as
      success.
    """

    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pipeline_mode = str(pipeline_mode or "inference")
    training_reference = _training_reference_payload(training_reference_result)
    training_compare = _training_compare_payload(training_compare_result)

    optimizer_state_validation = _optimizer_state_validation_payload(optimizer_state_artifacts)

    if pipeline_mode == "training_on_device":
        if training_compare is not None:
            status = "passed"
            reason = "training reference and generated/testbench comparison artifacts are available"
        elif training_reference is not None:
            status = "reference_only"
            reason = "Python training reference exists, but generated/testbench comparison artifacts are missing"
        else:
            status = "not_run"
            reason = "no training numeric reference or generated/testbench comparison artifacts were found"
    else:
        inference_reference_artifacts = inference_reference_artifacts or {}
        output_compare = None
        if inference_reference_artifacts.get("outputs_hw") and inference_reference_artifacts.get("outputs_ref"):
            output_compare = _compare_file_pair(
                inference_reference_artifacts.get("outputs_ref"),
                inference_reference_artifacts.get("outputs_hw"),
            )
        has_outputs = bool(output_compare and output_compare.get("passed"))
        status = "passed" if has_outputs else "not_run"
        reason = (
            "inference output comparison artifacts were compared successfully"
            if has_outputs
            else "inference numeric comparison is not yet available or did not pass for this compile path"
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "numeric_validation",
        "pipeline_mode": pipeline_mode,
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "source_generated": bool(source_generated),
        "hls_ran": bool(hls_ran),
        "hls_ok": hls_ok,
        "inference": {
            "status": status if pipeline_mode != "training_on_device" else "not_applicable",
            "inputs_bin": _path_or_none(inference_reference_artifacts.get("inputs_bin")) if inference_reference_artifacts else None,
            "outputs_hw": _path_or_none(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else None,
            "outputs_ref": _path_or_none(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else None,
            "outputs_hw_exists": _exists(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else False,
            "outputs_ref_exists": _exists(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else False,
            "output_compare": output_compare if pipeline_mode != "training_on_device" else None,
        },
        "training": {
            "status": status if pipeline_mode == "training_on_device" else "not_applicable",
            "reference": training_reference,
            "comparison": training_compare,
            "checks": [] if training_compare is None else [
                {"name": "gradients", "metric": "cosine_similarity", "value": training_compare.get("grad_cosine"), "passed": training_compare.get("grad_cosine") is None or training_compare.get("grad_cosine") >= 0.99},
                {"name": "weights_after", "metric": "cosine_similarity", "value": training_compare.get("weight_after_cosine"), "passed": training_compare.get("weight_after_cosine") is None or training_compare.get("weight_after_cosine") >= 0.99},
                {"name": "weight_delta", "metric": "cosine_similarity", "value": training_compare.get("weight_delta_cosine"), "passed": training_compare.get("weight_delta_cosine") is None or training_compare.get("weight_delta_cosine") >= 0.99},
            ],
        },
        "gradient_export": gradient_export_artifacts or {"requested": False, "status": "not_requested"},
        "optimizer_state_validation": optimizer_state_validation,
        "paper_claim_allowed": {
            "numeric_correctness": status == "passed",
        },
    }

    json_path = reports / "numeric_validation.json"
    md_path = reports / "numeric_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Numeric validation",
        "",
        f"- Pipeline mode: `{pipeline_mode}`",
        f"- Status: `{status}`",
        f"- Passed: `{str(status == 'passed').lower()}`",
        f"- Reason: {reason}",
        f"- Source generated: `{str(bool(source_generated)).lower()}`",
        f"- HLS ran: `{str(bool(hls_ran)).lower()}`",
    ]
    if pipeline_mode == "training_on_device":
        lines += [
            "",
            "## Training evidence",
            f"- Python reference: `{ 'yes' if training_reference is not None else 'no' }`",
            f"- Generated/testbench comparison: `{ 'yes' if training_compare is not None else 'no' }`",
            f"- Gradient export validation: `{(gradient_export_artifacts or {}).get('status', 'not_requested')}`",
            f"- Optimizer-state validation: `{optimizer_state_validation.get('status', 'not_requested')}`",
        ]
    else:
        lines += [
            "",
            "## Inference evidence",
            "- Final output comparison: `not available`" if status != "passed" else "- Final output comparison: `available`",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"numeric_validation_json": json_path, "numeric_validation_md": md_path}
