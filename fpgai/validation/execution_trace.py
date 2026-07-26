"""Stage-oriented localization for training numeric-equivalence failures.

This module converts tensor-level comparisons into a dependency-ordered
execution trace. It intentionally consumes existing semantic captures instead
of adding a second capture format. Future HLS intermediate probes can extend
``STAGE_SPECS`` without changing the report contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

STAGE_SPECS: tuple[dict[str, Any], ...] = (
    {"stage": "pre_update_forward", "roles": ("pre_update_loss",), "depends_on": ()},
    {"stage": "parameter_gradient", "roles": ("parameter_gradients",), "depends_on": ("pre_update_forward",)},
    {"stage": "optimizer_first_moment", "roles": ("optimizer_m_after",), "depends_on": ("parameter_gradient",)},
    {"stage": "optimizer_second_moment", "roles": ("optimizer_v_after",), "depends_on": ("parameter_gradient",)},
    {"stage": "optimizer_step", "roles": ("optimizer_step_after",), "depends_on": ()},
    {"stage": "parameter_update", "roles": ("weights_after", "biases_after"), "depends_on": ("optimizer_first_moment", "optimizer_second_moment")},
    {"stage": "post_update_forward", "roles": ("post_update_loss",), "depends_on": ("parameter_update",)},
)


def _role_status(comparison: Mapping[str, Any] | None) -> str:
    if not comparison:
        return "missing"
    if comparison.get("status") not in (None, "compared"):
        return str(comparison.get("status"))
    return "passed" if comparison.get("passed") is True else "failed"


def build_training_execution_trace(numeric_report: Mapping[str, Any], probe_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    comparisons = numeric_report.get("comparisons", {}) or {}
    stages: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None

    for spec in STAGE_SPECS:
        role_entries = []
        for role in spec["roles"]:
            comparison = comparisons.get(role)
            entry = {"role": role, "status": _role_status(comparison)}
            if comparison:
                entry.update({
                    "passed": comparison.get("passed"),
                    "max_abs_error": comparison.get("max_abs_error"),
                    "worst_mismatch": comparison.get("worst_mismatch"),
                })
            role_entries.append(entry)
        statuses = {entry["status"] for entry in role_entries}
        if "failed" in statuses:
            status = "failed"
        elif statuses == {"passed"}:
            status = "passed"
        elif "missing" in statuses:
            status = "incomplete"
        else:
            status = "not_comparable"
        stage = {
            "stage": spec["stage"],
            "status": status,
            "depends_on": list(spec["depends_on"]),
            "roles": role_entries,
        }
        stages.append(stage)
        if first_divergence is None and status == "failed":
            failed_role = next(entry for entry in role_entries if entry["status"] == "failed")
            first_divergence = {
                "stage": spec["stage"],
                "role": failed_role["role"],
                "max_abs_error": failed_role.get("max_abs_error"),
                "worst_mismatch": failed_role.get("worst_mismatch"),
            }

    if probe_report and probe_report.get("first_divergence"):
        first_divergence = {**probe_report["first_divergence"], "source": "intermediate_probe"}
    return {
        "artifact_kind": "fpgai_training_execution_trace",
        "schema_version": 1,
        "status": "diverged" if first_divergence else "passed",
        "source_numeric_status": numeric_report.get("status"),
        "first_divergence": first_divergence,
        "stages": stages,
        "interpretation": (
            "The first failed stage is the earliest divergence observable with the currently enabled semantic captures. "
            "Downstream failures are consequences until a lower-level intermediate probe demonstrates otherwise."
        ),
    }


def write_training_execution_trace(numeric_report_path: str | Path, output_path: str | Path, *, probe_comparison_path: str | Path | None = None) -> Path:
    source = Path(numeric_report_path)
    target = Path(output_path)
    probe_report = None
    if probe_comparison_path is not None and Path(probe_comparison_path).exists():
        probe_report = json.loads(Path(probe_comparison_path).read_text(encoding="utf-8"))
    payload = build_training_execution_trace(json.loads(source.read_text(encoding="utf-8")), probe_report)
    payload["source_numeric_report"] = str(source)
    if probe_comparison_path is not None:
        payload["source_probe_comparison"] = str(probe_comparison_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
