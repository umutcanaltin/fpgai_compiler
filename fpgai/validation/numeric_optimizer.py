"""Optimizer state, parameter update, latency, and resource validation."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from fpgai.analysis.hls_estimate_compare import parse_hls_csynth_report

from .numeric_common import (
    _cfg_lookup,
    _compare_file_pair,
    _compare_vectors,
    _copy_or_transform_f32,
    _exists,
    _normalize_sequence_entries,
    _path_or_none,
    _positive_int,
    _read_f32_file,
    _read_json_file,
)

def _optimizer_state_lsb(raw_config: dict[str, Any] | None) -> float:
    """Resolve the optimizer-state fixed-point LSB from the numeric contract."""
    raw = raw_config or {}
    spec = _cfg_lookup(raw, "numerics.training.optimizer_state", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, "numerics.defaults.accum", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, "precision.defaults.accum", None)
    if not isinstance(spec, dict):
        return float(2.0 ** -16)
    try:
        total_bits = int(spec.get("total_bits", spec.get("bits", 24)))
        int_bits = int(spec.get("int_bits", spec.get("integer_bits", 8)))
    except Exception:
        return float(2.0 ** -16)
    return float(2.0 ** (-max(0, total_bits - int_bits)))


def _optimizer_state_reference_metadata(ref_path: Any) -> dict[str, Any]:
    """Load optimizer update count and canonical parameter layout beside a reference artifact."""
    try:
        ref = Path(ref_path)
    except Exception:
        return {}
    root = ref.parent
    payload: dict[str, Any] = {}
    candidate_roots = [root, root.parent]
    for candidate_root in candidate_roots:
        summary_path = candidate_root / "training_hardware_domain_reference.json"
        try:
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                payload["optimizer_updates"] = int(summary.get("optimizer_updates", 0) or 0)
                payload["optimizer_type"] = str(summary.get("optimizer_type", "sgd"))
                break
        except Exception:
            pass
    for candidate_root in candidate_roots:
        layout_path = candidate_root / "per_sample_trace" / "parameter_layer_map.json"
        try:
            if layout_path.exists():
                layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
                entries = layout_payload.get("entries", [])
                if isinstance(entries, list):
                    payload["parameter_layout"] = [entry for entry in entries if isinstance(entry, dict)]
                    break
        except Exception:
            pass
    return payload


def _optimizer_state_segments(
    *,
    optimizer: str,
    parameter_layout: list[dict[str, Any]],
    total_words: int,
) -> list[dict[str, Any]]:
    """Build canonical packed optimizer-state segments from the existing parameter layer map."""
    parameter_words = sum(int(entry.get("count", 0) or 0) for entry in parameter_layout)
    if parameter_words <= 0:
        return []
    segments: list[dict[str, Any]] = []
    groups = (("velocity", 0),) if optimizer == "momentum" else (("m", 0), ("v", parameter_words))
    for group, base in groups:
        for entry in parameter_layout:
            count = int(entry.get("count", 0) or 0)
            offset = base + int(entry.get("offset", 0) or 0)
            layer = str(entry.get("layer", "parameter"))
            role = str(entry.get("role", "state"))
            role_prefix = {"weight": "W", "bias": "B", "gamma": "gamma", "beta": "beta"}.get(role, role)
            segments.append({
                "name": f"{group}_{role_prefix}_{layer}",
                "offset": offset,
                "count": count,
                "layer": layer,
                "role": role,
            })
    if optimizer == "adam" and total_words == (2 * parameter_words + 1):
        segments.append({"name": "step", "offset": 2 * parameter_words, "count": 1, "layer": None, "role": "optimizer_step"})
    return segments


def _compare_optimizer_state_pair(
    ref_path: Any,
    got_path: Any,
    *,
    lsb: float,
    optimizer: str = "sgd",
) -> dict[str, Any]:
    """Compare canonical optimizer state with strict and propagated fixed-point classifications."""
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    metadata = _optimizer_state_reference_metadata(ref_path)
    optimizer_updates = int(metadata.get("optimizer_updates", 0) or 0)
    allowed_lsb = 1 if optimizer_updates <= 1 else 2
    payload: dict[str, Any] = {
        "ref_path": _path_or_none(ref_path),
        "got_path": _path_or_none(got_path),
        "ref_exists": _exists(ref_path),
        "got_exists": _exists(got_path),
        "optimizer_updates": optimizer_updates or None,
        "numeric_tolerance": {
            "kind": "optimizer_state_lsb",
            "lsb": float(lsb),
            "single_update_allowed_lsb": 1,
            "multi_update_allowed_lsb": 2,
            "allowed_lsb": allowed_lsb,
        },
    }
    if ref is None or got is None:
        payload.update({"status": "artifact_missing", "passed": False})
        return payload
    metrics = _compare_vectors(ref, got)
    n = min(len(ref), len(got))
    diffs = [float(got[i] - ref[i]) for i in range(n)]
    abs_diffs = [abs(value) for value in diffs]
    ref_norm = math.sqrt(sum(float(value) * float(value) for value in ref))
    diff_norm = math.sqrt(sum(value * value for value in diffs))
    exact_words = sum(1 for value in diffs if value == 0.0)
    eps = 1.0e-15
    within_one_lsb = sum(1 for value in abs_diffs if value <= float(lsb) + eps)
    within_two_lsb = sum(1 for value in abs_diffs if value <= (2.0 * float(lsb)) + eps)
    above_two_lsb = sum(1 for value in abs_diffs if value > (2.0 * float(lsb)) + eps)
    same_shape = len(ref) == len(got)
    all_within_one = bool(same_shape and within_one_lsb == len(ref))
    all_within_two = bool(same_shape and within_two_lsb == len(ref))
    passed = all_within_one if allowed_lsb == 1 else all_within_two
    classification = (
        "bit_exact" if same_shape and exact_words == len(ref)
        else "quantization_aligned" if all_within_one
        else "propagated_quantization_aligned" if optimizer_updates > 1 and all_within_two
        else "failed"
    )
    maximum_lsb_distance = max((int(round(value / float(lsb))) for value in abs_diffs), default=0) if lsb > 0 else None

    segment_payloads: list[dict[str, Any]] = []
    for segment in _optimizer_state_segments(
        optimizer=optimizer,
        parameter_layout=metadata.get("parameter_layout", []),
        total_words=len(ref),
    ):
        start = int(segment["offset"])
        end = start + int(segment["count"])
        segment_diffs = abs_diffs[start:end]
        segment_payloads.append({
            **segment,
            "exact_words": sum(1 for value in segment_diffs if value == 0.0),
            "within_one_lsb": sum(1 for value in segment_diffs if value <= float(lsb) + eps),
            "within_two_lsb": sum(1 for value in segment_diffs if value <= (2.0 * float(lsb)) + eps),
            "above_two_lsb": sum(1 for value in segment_diffs if value > (2.0 * float(lsb)) + eps),
            "max_abs_error": max(segment_diffs, default=0.0),
            "maximum_lsb_distance": max((int(round(value / float(lsb))) for value in segment_diffs), default=0) if lsb > 0 else None,
        })

    payload.update(metrics)
    payload.update({
        "reference_words": len(ref),
        "hls_words": len(got),
        "reference_nonzero": sum(1 for value in ref if value != 0.0),
        "hls_nonzero": sum(1 for value in got if value != 0.0),
        "reference_norm": ref_norm,
        "hls_norm": math.sqrt(sum(float(value) * float(value) for value in got)),
        "relative_l2": (diff_norm / ref_norm) if ref_norm > 0.0 else diff_norm,
        "exact_words": exact_words,
        "within_one_lsb": within_one_lsb,
        "within_two_lsb": within_two_lsb,
        "above_two_lsb": above_two_lsb,
        "all_words_within_one_lsb": all_within_one,
        "all_words_within_two_lsb": all_within_two,
        "maximum_lsb_distance": maximum_lsb_distance,
        "classification": classification,
        "segments": segment_payloads,
        "status": "compared" if same_shape else "shape_mismatch",
        "passed": passed,
    })
    return payload


def _optimizer_state_validation_payload(optimizer_state_artifacts: dict[str, Any] | None, *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe and compare persistent optimizer-state tensors when artifacts exist.

    Momentum and Adam correctness is not only weights-after correctness: their
    persistent state must also be checked.  This helper records requested state
    tensors and compares explicit ref/got float32 files when the testbench or
    runtime path provides them.  Missing files are reported as missing artifacts,
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
    optimizer = str(payload.get("optimizer", "sgd")).lower().replace("-", "_")

    if isinstance(comparisons_cfg, dict):
        for name, cfg in sorted(comparisons_cfg.items()):
            if not isinstance(cfg, dict):
                continue
            cmp_payload = _compare_optimizer_state_pair(
                cfg.get("ref"), cfg.get("got"), lsb=_optimizer_state_lsb(raw_config), optimizer=optimizer
            )
            comparisons[str(name)] = cmp_payload
            if cmp_payload.get("status") == "compared":
                any_compared = True
            if not bool(cmp_payload.get("passed", False)):
                all_passed = False

    payload["comparisons"] = comparisons
    if optimizer not in {"momentum", "adam"}:
        payload["status"] = "not_applicable"
        payload["implementation_status"] = "not_applicable"
        payload["passed"] = False
    elif not requested:
        payload["status"] = "not_validated"
        payload["implementation_status"] = "not_validated"
        payload["passed"] = False
    elif comparisons and any_compared and all_passed:
        legacy_comparison_only = "layout" not in payload and not any(
            key in payload for key in ("layout_version", "reference_domain", "claim_scope")
        )
        payload["status"] = "compared" if legacy_comparison_only else "implemented"
        payload["implementation_status"] = "implemented"
        payload["passed"] = True
    elif comparisons:
        both_sides_seen = any(
            bool(cmp_payload.get("ref_exists", False)) and bool(cmp_payload.get("got_exists", False))
            for cmp_payload in comparisons.values()
            if isinstance(cmp_payload, dict)
        )
        any_ref_missing_got = any(
            bool(cmp_payload.get("ref_exists", False)) and not bool(cmp_payload.get("got_exists", False))
            for cmp_payload in comparisons.values()
            if isinstance(cmp_payload, dict)
        )
        original_status = str(payload.get("status") or "")
        if both_sides_seen:
            payload["status"] = "failed"
            payload["implementation_status"] = "failed"
            payload["artifact_status"] = "failed"
        elif any_ref_missing_got:
            # Keep the generated-export capability status for legacy contract
            # reports, but expose the stricter numeric proof status separately.
            # Non-export optimizer-state validation remains artifact_missing.
            payload["artifact_status"] = "artifact_missing"
            if original_status == "generated_export_capture_supported":
                payload["status"] = original_status
            else:
                payload["status"] = "artifact_missing"
                payload["implementation_status"] = "not_validated"
        else:
            payload["status"] = "not_validated"
            payload["implementation_status"] = "not_validated"
            payload.setdefault("artifact_status", "artifact_missing")
        payload["passed"] = False
    else:
        payload["status"] = "not_validated" if requested else "not_applicable"
        payload["implementation_status"] = payload["status"]
        payload["passed"] = False
    return payload


def _parameter_role_lsb(raw_config: dict[str, Any] | None, role: str) -> float:
    raw = raw_config or {}
    role_key = "weight" if role == "weight" else "bias" if role == "bias" else "weight"
    spec = _cfg_lookup(raw, f"numerics.defaults.{role_key}", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, f"precision.defaults.{role_key}", None)
    if not isinstance(spec, dict):
        defaults = {"weight": (20, 8), "bias": (32, 16)}
        total_bits, int_bits = defaults.get(role_key, (20, 8))
    else:
        total_bits = int(spec.get("total_bits", spec.get("bits", 20 if role_key == "weight" else 32)))
        int_bits = int(spec.get("int_bits", spec.get("integer_bits", 8 if role_key == "weight" else 16)))
    return float(2.0 ** (-max(0, total_bits - int_bits)))


def _parameter_update_validation_payload(parameter_artifacts: dict[str, Any] | None, *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not parameter_artifacts:
        return {"requested": False, "status": "not_requested", "implementation_status": "not_requested", "passed": False}
    payload = dict(parameter_artifacts)
    ref_path = payload.get("ref")
    got_path = payload.get("got")
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    payload.update({"ref_path": _path_or_none(ref_path), "got_path": _path_or_none(got_path), "ref_exists": _exists(ref_path), "got_exists": _exists(got_path)})
    if ref is None or got is None:
        payload.update({"status": "artifact_missing", "implementation_status": "not_validated", "passed": False})
        return payload
    metadata = _optimizer_state_reference_metadata(ref_path)
    updates = int(metadata.get("optimizer_updates", 0) or 0)
    layout = metadata.get("parameter_layout", [])
    same_shape = len(ref) == len(got)
    segments = []
    all_passed = same_shape
    exact_total = within1_total = within2_total = above2_total = 0
    max_distance = 0
    eps = 1e-15
    for entry in layout:
        start = int(entry.get("offset", 0) or 0); count = int(entry.get("count", 0) or 0); end = start + count
        role = str(entry.get("role", "weight")); lsb = _parameter_role_lsb(raw_config, role)
        diffs = [abs(float(got[i] - ref[i])) for i in range(start, min(end, len(ref), len(got)))]
        exact = sum(v == 0.0 for v in diffs); within1 = sum(v <= lsb + eps for v in diffs); within2 = sum(v <= 2*lsb + eps for v in diffs); above2 = sum(v > 2*lsb + eps for v in diffs)
        allowed = 1 if updates <= 1 else 2
        passed = len(diffs) == count and (within1 == count if allowed == 1 else within2 == count)
        all_passed = all_passed and passed
        distance = max((int(round(v/lsb)) for v in diffs), default=0) if lsb > 0 else 0
        max_distance = max(max_distance, distance)
        exact_total += exact; within1_total += within1; within2_total += within2; above2_total += above2
        segments.append({"name": f"{'W' if role == 'weight' else 'B' if role == 'bias' else role}_{entry.get('layer','parameter')}", "layer": entry.get("layer"), "role": role, "offset": start, "count": count, "lsb": lsb, "exact_words": exact, "within_one_lsb": within1, "within_two_lsb": within2, "above_two_lsb": above2, "maximum_lsb_distance": distance, "max_abs_error": max(diffs, default=0.0), "passed": passed})
    metrics = _compare_vectors(ref, got)
    classification = "bit_exact" if same_shape and exact_total == len(ref) else "quantization_aligned" if same_shape and within1_total == len(ref) else "propagated_quantization_aligned" if updates > 1 and same_shape and within2_total == len(ref) else "failed"
    payload.update(metrics)
    payload.update({"optimizer_updates": updates or None, "reference_words": len(ref), "hls_words": len(got), "exact_words": exact_total, "within_one_lsb": within1_total, "within_two_lsb": within2_total, "above_two_lsb": above2_total, "all_words_within_one_lsb": bool(same_shape and within1_total == len(ref)), "all_words_within_two_lsb": bool(same_shape and within2_total == len(ref)), "maximum_lsb_distance": max_distance, "classification": classification, "segments": segments, "status": "implemented" if all_passed else "failed", "implementation_status": "implemented" if all_passed else "failed", "passed": all_passed})
    return payload


def _training_update_behavior_payload(parameter_validation: dict[str, Any], optimizer_validation: dict[str, Any], *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    parameter_segments = parameter_validation.get("segments", []) or []
    optimizer_comparisons = optimizer_validation.get("comparisons", {}) or {}
    optimizer_after = optimizer_comparisons.get("packed_optimizer_state_after", {}) if isinstance(optimizer_comparisons, dict) else {}
    optimizer_segments = optimizer_after.get("segments", []) or []
    first_parameter = next((segment for segment in parameter_segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
    first_optimizer = next((segment for segment in optimizer_segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
    updates = int(parameter_validation.get("optimizer_updates") or optimizer_after.get("optimizer_updates") or 0)
    per_update: list[dict[str, Any]] = []
    ref_path = Path(str(parameter_validation.get("ref_path"))) if parameter_validation.get("ref_path") else None
    got_path = Path(str(parameter_validation.get("got_path"))) if parameter_validation.get("got_path") else None
    ref_trace = ref_path.parent / "per_update_trace" if ref_path is not None else None
    got_trace = got_path.parent / "per_update_trace" if got_path is not None else None
    optimizer = str(optimizer_validation.get("optimizer", "sgd"))
    if ref_trace is not None and got_trace is not None and ref_trace.exists() and got_trace.exists():
        for update_index in range(1, updates + 1):
            weight_ref = ref_trace / f"update_{update_index:04d}_weights_ref.bin"
            weight_got = got_trace / f"update_{update_index:04d}_weights.bin"
            state_ref = ref_trace / f"update_{update_index:04d}_optimizer_state_ref.bin"
            state_got = got_trace / f"update_{update_index:04d}_optimizer_state.bin"
            weight_cmp = _parameter_update_validation_payload({"requested": True, "ref": weight_ref, "got": weight_got}, raw_config=raw_config)
            state_cmp = _compare_optimizer_state_pair(state_ref, state_got, lsb=_optimizer_state_lsb(raw_config), optimizer=optimizer) if state_ref.exists() or state_got.exists() else {"status": "not_applicable", "passed": optimizer == "sgd", "classification": "bit_exact" if optimizer == "sgd" else "not_applicable"}
            per_update.append({"update": update_index, "weights": weight_cmp, "optimizer_state": state_cmp, "passed": bool(weight_cmp.get("passed")) and bool(state_cmp.get("passed", optimizer == "sgd"))})

    first_divergent_update = None
    first_divergent_layer = None
    first_divergent_tensor = None
    for row in per_update:
        weight_cmp = row.get("weights", {}) or {}
        state_cmp = row.get("optimizer_state", {}) or {}
        weight_non_exact = weight_cmp.get("classification") not in (None, "bit_exact")
        state_non_exact = state_cmp.get("classification") not in (None, "bit_exact", "not_applicable")
        if weight_non_exact or state_non_exact:
            first_divergent_update = row.get("update")
            source = weight_cmp if weight_non_exact else state_cmp
            segments = source.get("segments", []) or []
            boundary = next((segment for segment in segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
            if isinstance(boundary, dict):
                first_divergent_layer = boundary.get("layer")
                first_divergent_tensor = boundary.get("name")
            else:
                first_divergent_tensor = "weights" if weight_non_exact else "optimizer_state"
            break

    if first_divergent_update is None and isinstance(first_parameter, dict):
        first_divergent_update = 1
        first_divergent_layer = first_parameter.get("layer")
        first_divergent_tensor = first_parameter.get("name")
    elif first_divergent_update is None and isinstance(first_optimizer, dict):
        first_divergent_update = 2 if updates and updates > 1 else 1
        first_divergent_layer = first_optimizer.get("layer")
        first_divergent_tensor = first_optimizer.get("name")
    if first_divergent_layer is None and isinstance(first_divergent_tensor, str) and "_" in first_divergent_tensor:
        first_divergent_layer = first_divergent_tensor.split("_", 1)[1]

    complete_trace = bool(updates > 0 and len(per_update) == updates)
    final_passed = bool(parameter_validation.get("passed") and optimizer_validation.get("passed"))
    status = "implemented" if final_passed and (complete_trace or not per_update) else "partial"
    return {
        "artifact_kind": "fpgai_training_update_behavior_trace",
        "schema_version": 3,
        "status": status,
        "optimizer_updates": updates or None,
        "per_update_trace_available": complete_trace,
        "per_update": per_update,
        "first_divergent_update": first_divergent_update,
        "first_divergent_layer": first_divergent_layer,
        "first_divergent_tensor": first_divergent_tensor,
        "first_parameter_boundary": first_parameter,
        "first_optimizer_state_boundary": first_optimizer,
        "propagation_path": [value for value in [first_parameter.get("name") if isinstance(first_parameter, dict) else None, first_optimizer.get("name") if isinstance(first_optimizer, dict) else None] if value],
        "final_classification": optimizer_after.get("classification") or parameter_validation.get("classification"),
        "note": "The first divergence is the first non-bit-exact per-update boundary, even when the comparison remains within the accepted fixed-point tolerance."
    }


def _optimizer_csynth_metrics(report_path: str | Path | None) -> dict[str, Any]:
    if report_path is None:
        return {}
    path = Path(report_path)
    if not path.is_file():
        return {}
    metrics = dict(parse_hls_csynth_report(path))
    text = path.read_text(encoding="utf-8", errors="ignore")
    xml_candidates = [path.with_suffix(".xml"), path.parent / "csynth.xml"]
    for candidate in xml_candidates:
        if candidate.is_file():
            text += "\n" + candidate.read_text(encoding="utf-8", errors="ignore")
    def first(patterns: list[str]) -> float | None:
        import re
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(str(match.group(1)).replace(",", ""))
                except Exception:
                    pass
        return None
    metrics.update({
        "actual_uram": first([r"<URAM>([0-9,.]+)</URAM>", r"\bURAM\b[^0-9\n]*([0-9,]+)"]),
        "latency_min_cycles": first([r"<Best-caseLatency>([0-9,.]+)</Best-caseLatency>", r"Latency[^\n]*min[^0-9\n]*([0-9,]+)"]),
        "latency_max_cycles": first([r"<Worst-caseLatency>([0-9,.]+)</Worst-caseLatency>", r"Latency[^\n]*max[^0-9\n]*([0-9,]+)"]),
        "interval_min_cycles": first([r"<Interval-min>([0-9,.]+)</Interval-min>", r"Interval[^\n]*min[^0-9\n]*([0-9,]+)"]),
        "interval_max_cycles": first([r"<Interval-max>([0-9,.]+)</Interval-max>", r"Interval[^\n]*max[^0-9\n]*([0-9,]+)"]),
        "estimated_clock_period_ns": first([r"<EstimatedClockPeriod>([0-9,.]+)</EstimatedClockPeriod>", r"Estimated[^\n]*Clock[^0-9\n]*([0-9.]+)"]),
    })
    return metrics


def _training_command_latency_payload(
    raw_config: dict[str, Any] | None,
    *,
    source_path: Path | None,
    hls_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Describe command-specific latency boundaries without inventing HLS results.

    Vitis HLS reports one aggregate latency range for the multi-command top.
    FPGAI therefore records that range separately and derives only source-level
    transfer lower bounds for export commands. These lower bounds are not
    presented as scheduled or implemented latency.
    """
    import re

    raw = raw_config or {}
    text = ""
    if source_path is not None and source_path.is_file():
        text = source_path.read_text(encoding="utf-8", errors="ignore")

    weight_words = 0
    for _type_name, _name, count in re.findall(
        r"static\s+(wgt_t|bias_t)\s+([WB]_[A-Za-z0-9_]+)\[(\d+)\]", text
    ):
        weight_words += int(count)
    optimizer_words = None
    match = re.search(r"FPGAI_OPTIMIZER_STATE_EXPORT_WORDS\s*=\s*(\d+)", text)
    if match:
        optimizer_words = int(match.group(1))

    materialization = str(_cfg_lookup(raw, "training.gradients.materialization", "full") or "full")
    tile_size = int(_cfg_lookup(raw, "training.gradients.tile_size", 256) or 256)
    lifetime = str(_cfg_lookup(raw, "training.memory_lifetime.policy", "separate") or "separate")

    aggregate = {
        "latency_min_cycles": hls_metrics.get("latency_min_cycles"),
        "latency_max_cycles": hls_metrics.get("latency_max_cycles"),
        "interval_min_cycles": hls_metrics.get("interval_min_cycles"),
        "interval_max_cycles": hls_metrics.get("interval_max_cycles"),
        "estimated_clock_period_ns": hls_metrics.get("estimated_clock_period_ns"),
        "scope": "multi_command_top_aggregate",
    }
    commands = {
        "run_training": {
            "status": "aggregate_hls_range_only",
            "hls_top_range": aggregate,
            "note": "The current C-synthesis report does not separate runtime command branches.",
        },
        "export_weights": {
            "status": "source_transfer_lower_bound",
            "output_words": weight_words or None,
            "minimum_stream_cycles": weight_words or None,
        },
        "export_gradients": {
            "status": "source_transfer_lower_bound",
            "output_words": weight_words or None,
            "minimum_stream_cycles": weight_words or None,
            "materialization": materialization,
            "tile_size": tile_size if materialization == "tiled" else None,
            "memory_lifetime_policy": lifetime,
        },
        "export_optimizer_state": {
            "status": "source_transfer_lower_bound" if optimizer_words else "not_available",
            "output_words": optimizer_words,
            "minimum_transfer_cycles": optimizer_words,
        },
    }
    return {
        "artifact_kind": "fpgai_training_command_latency",
        "schema_version": 1,
        "status": "implemented" if hls_metrics else "not_validated",
        "aggregate_hls_top": aggregate,
        "commands": commands,
        "claim_scope": "aggregate_hls_range_plus_source_transfer_lower_bounds",
        "note": "Source transfer counts are lower bounds, not command-specific scheduled latency.",
    }


def _training_resource_owner_payload(
    raw_config: dict[str, Any] | None,
    *,
    source_path: Path | None,
    hls_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Describe generated training-memory owners and the YAML knobs that own them.

    Array block counts are conservative source-bound lower-bound estimates. The
    canonical device totals remain the parsed C-synthesis metrics.
    """
    raw = raw_config or {}
    import math
    import re

    type_bits = {
        "wgt_t": int(_cfg_lookup(raw, "numerics.weights.total_bits", 16) or 16),
        "bias_t": int(_cfg_lookup(raw, "numerics.bias.total_bits", 16) or 16),
        "act_t": int(_cfg_lookup(raw, "numerics.activations.total_bits", 16) or 16),
        "grad_wgt_t": int(_cfg_lookup(raw, "numerics.gradients.total_bits", 16) or 16),
        "grad_bias_t": int(_cfg_lookup(raw, "numerics.gradients.total_bits", 16) or 16),
        "opt_t": int(_cfg_lookup(raw, "numerics.optimizer_state.total_bits", 32) or 32),
        "float": 32,
    }
    owners: list[dict[str, Any]] = []
    if source_path is not None and source_path.is_file():
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        macro_counts = {
            name: int(value)
            for name, value in re.findall(r"#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)", text)
        }
        decl = re.compile(
            r"(?:static\s+)?(?P<type>wgt_t|bias_t|act_t|grad_wgt_t|grad_bias_t|opt_t|float)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<count>\d+|[A-Za-z_][A-Za-z0-9_]*)\s*\]\s*;"
        )
        for match in decl.finditer(text):
            type_name = match.group("type")
            name = match.group("name")
            count_token = match.group("count")
            count = int(count_token) if count_token.isdigit() else macro_counts.get(count_token)
            if count is None:
                continue
            bind = re.search(
                rf"#pragma\s+HLS\s+BIND_STORAGE\s+variable={re.escape(name)}\s+type=ram_[12]p\s+impl=(bram|uram)",
                text,
            )
            storage = bind.group(1) if bind else None
            bits = count * type_bits.get(type_name, 32)
            if storage == "uram":
                lower_bound_blocks = int(math.ceil(bits / 294912.0))
            elif storage == "bram":
                lower_bound_blocks = int(math.ceil(bits / 18432.0))
            else:
                lower_bound_blocks = None
            upper = name.upper()
            if upper.startswith("FPGAI_ADAM_") or upper.startswith("FPGAI_MOMENTUM_"):
                role = "optimizer_state"
                knob = "training.storage.optimizer_state"
            elif "GRADIENT_EXPORT_TILE" in upper:
                role = "gradient_export_scratch"
                knob = "training.gradients.tile_size"
            elif upper.startswith("DW_") or upper.startswith("DB_") or "DW_TILE" in upper or "GRAD" in upper:
                role = "gradient"
                knob = "training.storage.parameter_gradient"
            elif upper.startswith("W_") or upper.startswith("B_"):
                role = "parameter"
                knob = "memory.weight_storage"
            elif "TILE" in upper or "BUFFER" in upper or type_name == "act_t":
                role = "activation_or_scratch"
                knob = "data_movement.*.tiled.tile_size"
            else:
                role = "other"
                knob = None
            owners.append({
                "name": name,
                "role": role,
                "element_type": type_name,
                "element_bits": type_bits.get(type_name, 32),
                "count": count,
                "total_bits": bits,
                "source_binding": storage,
                "source_bound_lower_bound_blocks": lower_bound_blocks,
                "owning_yaml_knob": knob,
            })
    owners.sort(key=lambda row: int(row.get("total_bits") or 0), reverse=True)
    knob_trace = [
        {"path": "optimization.parallel.pe", "effective": _cfg_lookup(raw, "optimization.parallel.pe", None), "effect": "output/channel parallelism"},
        {"path": "optimization.parallel.simd", "effective": _cfg_lookup(raw, "optimization.parallel.simd", None), "effect": "input/channel parallelism"},
        {"path": "optimization.parallel.unroll_factor", "effective": _cfg_lookup(raw, "optimization.parallel.unroll_factor", None), "effect": "loop replication"},
        {"path": "optimization.parallel.partition_factor", "effective": _cfg_lookup(raw, "optimization.parallel.partition_factor", None), "effect": "array banking and memory replication"},
        {"path": "optimization.parallel.array_partition_mode", "effective": _cfg_lookup(raw, "optimization.parallel.array_partition_mode", None), "effect": "banking layout"},
        {"path": "memory.weight_storage", "effective": _cfg_lookup(raw, "memory.weight_storage", None), "effect": "parameter memory implementation"},
        {"path": "training.gradients.computation", "effective": _cfg_lookup(raw, "training.gradients.computation", "full_buffer"), "effect": "parameter-gradient compute lifetime: full buffer, tiled accumulation, or fused update"},
        {"path": "training.storage.parameter_gradient", "effective": _cfg_lookup(raw, "training.storage.parameter_gradient", _cfg_lookup(raw, "training.storage.gradient", _cfg_lookup(raw, "memory.gradient_storage", None))), "effect": "parameter-gradient memory implementation"},
        {"path": "training.storage.optimizer_state", "effective": _cfg_lookup(raw, "training.storage.optimizer_state", _cfg_lookup(raw, "memory.optimizer_state_storage", None)), "effect": "optimizer-state memory implementation"},
        {"path": "training.gradients.materialization", "effective": _cfg_lookup(raw, "training.gradients.materialization", "full"), "effect": "full, tiled, or streamed gradient export scratch structure"},
        {"path": "training.gradients.tile_size", "effective": _cfg_lookup(raw, "training.gradients.tile_size", 256), "effect": "bounded gradient export tile size"},
        {"path": "training.memory_lifetime.policy", "effective": _cfg_lookup(raw, "training.memory_lifetime.policy", "separate"), "effect": "per-layer or phase-shared physical export tile ownership"},
        {"path": "training.optimizer.implementation.arithmetic", "effective": _cfg_lookup(raw, "training.optimizer.implementation.arithmetic", "direct"), "effect": "optimizer arithmetic ownership"},
        {"path": "training.optimizer.implementation.update_parallelism", "effective": _cfg_lookup(raw, "training.optimizer.implementation.update_parallelism", 1), "effect": "optimizer update replication"},
        {"path": "data_movement.gradients.export.tiled.tile_size", "effective": _cfg_lookup(raw, "data_movement.gradients.export.tiled.tile_size", _cfg_lookup(raw, "data_movement.gradients.export.tile_size", None)), "effect": "gradient export scratch tile"},
        {"path": "data_movement.inputs.import.tiled.tile_size", "effective": _cfg_lookup(raw, "data_movement.inputs.import.tiled.tile_size", _cfg_lookup(raw, "data_movement.inputs.import.tile_size", None)), "effect": "input activation scratch tile"},
    ]
    actual_bram = hls_metrics.get("actual_bram18")
    actual_lut = hls_metrics.get("actual_lut")
    recommendations = []
    if isinstance(actual_bram, (int, float)) and actual_bram > 0:
        recommendations.extend([
            {"priority": 1, "resource": "bram_18k", "knob": "optimization.parallel.partition_factor", "action": "keep at 1 or reduce banking", "reason": "partitioning can replicate memory banks"},
            {"priority": 2, "resource": "bram_18k", "knob": "training.storage.optimizer_state", "action": "prefer uram when supported", "reason": "persistent optimizer state is a dominant owner"},
            {"priority": 3, "resource": "bram_18k", "knob": "training.storage.parameter_gradient", "action": "select uram when capacity permits", "reason": "moves complete dW/dB owners from BRAM to URAM"},
            {"priority": 4, "resource": "bram_18k", "knob": "training.gradients.materialization", "action": "select tiled or streamed", "reason": "removes full OUT_grad export scratch arrays"},
            {"priority": 5, "resource": "bram_18k", "knob": "training.memory_lifetime.policy", "action": "select phase_shared with tiled materialization", "reason": "reuses one physical export tile across layers"},
        ])
    recommendations.extend([
        {"priority": 7, "resource": "lut", "knob": "optimization.parallel.pe", "action": "reduce", "reason": "reduces replicated datapaths"},
        {"priority": 5, "resource": "lut", "knob": "optimization.parallel.simd", "action": "reduce", "reason": "reduces replicated lane logic"},
        {"priority": 6, "resource": "lut", "knob": "training.optimizer.implementation.update_parallelism", "action": "set to 1", "reason": "serializes optimizer update lanes"},
    ])
    return {
        "artifact_kind": "fpgai_training_resource_ownership",
        "schema_version": 1,
        "status": "implemented" if source_path is not None else "not_available",
        "source": str(source_path) if source_path is not None else None,
        "canonical_hls_totals": hls_metrics,
        "owner_count": len(owners),
        "owners": owners,
        "top_owners": owners[:20],
        "hardware_knob_trace": knob_trace,
        "recommended_knob_actions": recommendations,
        "claim_scope": "generated_source_ownership_with_hls_device_totals",
        "note": "Per-array block values are source-bound lower-bound estimates; canonical totals come from C synthesis.",
    }


def _optimizer_resource_strategy_payload(
    raw_config: dict[str, Any] | None,
    *,
    hls_ran: bool,
    hls_ok: bool | None,
    hls_csynth_report: str | Path | None = None,
) -> dict[str, Any]:
    raw = raw_config or {}
    arithmetic = str(_cfg_lookup(raw, "training.optimizer.implementation.arithmetic", "direct") or "direct").lower().replace("-", "_")
    parallelism = int(_cfg_lookup(raw, "training.optimizer.implementation.update_parallelism", 1) or 1)
    storage = str(_cfg_lookup(raw, "memory.optimizer_state_storage", _cfg_lookup(raw, "training.storage.optimizer_state", "none")) or "none").lower()
    shared_requested = arithmetic == "shared"
    report_path = Path(hls_csynth_report) if hls_csynth_report else None
    report_present = bool(report_path and report_path.is_file())
    metrics = _optimizer_csynth_metrics(report_path) if report_present else {}
    synthesis_status = "available" if hls_ran and hls_ok is True and report_present else ("failed" if hls_ran and hls_ok is False else "not_validated")
    state_impl = "uram" if storage == "uram" else ("bram" if storage == "bram" else None)
    source_path = None
    state_array_bindings: list[dict[str, Any]] = []
    if report_path is not None:
        for parent in [report_path.parent, *report_path.parents]:
            candidate = parent / "src" / "deeplearn.cpp"
            if candidate.is_file():
                source_path = candidate
                break
    if source_path is not None:
        import re
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"static\s+opt_t\s+(FPGAI_ADAM_[MV]_[WB]_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*;", source_text):
            name = match.group(1)
            size = int(match.group(2))
            bind = re.search(rf"#pragma\s+HLS\s+BIND_STORAGE\s+variable={re.escape(name)}\s+type=ram_2p\s+impl=(bram|uram)", source_text)
            state_array_bindings.append({"name": name, "count": size, "requested_storage": storage, "source_binding": bind.group(1) if bind else None, "binding_status": "materialized" if bind else "missing"})
    synthesized_storage_status = "not_validated"
    if synthesis_status == "available":
        actual_uram = metrics.get("actual_uram")
        if storage == "uram":
            synthesized_storage_status = "observed" if actual_uram not in (None, 0, 0.0) else "not_observed"
        elif storage == "bram":
            synthesized_storage_status = "observed" if metrics.get("actual_bram18") not in (None, 0, 0.0) else "not_observed"
    return {
        "artifact_kind": "fpgai_optimizer_resource_strategy",
        "schema_version": 2,
        "arithmetic": arithmetic,
        "update_parallelism": parallelism,
        "optimizer_state_storage": storage,
        "mechanism": "single_non_inlined_adam_correction_owner" if shared_requested else "direct_per_update_loop_expression",
        "source_generation_status": "implemented",
        "hls_synthesis_status": synthesis_status,
        "hls_ran": bool(hls_ran),
        "hls_ok": hls_ok,
        "csynth_report": str(report_path) if report_path is not None else None,
        "csynth_report_present": report_present,
        "hls_metrics": metrics,
        "state_array_bindings": state_array_bindings,
        "state_binding_source": str(source_path) if source_path is not None else None,
        "training_resource_ownership": _training_resource_owner_payload(raw, source_path=source_path, hls_metrics=metrics),
        "training_command_latency": _training_command_latency_payload(raw, source_path=source_path, hls_metrics=metrics),
        "requested_state_impl": state_impl,
        "synthesized_storage_status": synthesized_storage_status,
        "baseline_comparison_status": "not_run",
        "claim_scope": "generated_and_synthesized_strategy" if synthesis_status == "available" else "generated_strategy",
    }
