"""Locate native hls4ml CSim outputs and audit all selected predictions.

The hls4ml 1.3.0 generated CSim testbench prints "Processing input"
only every 5000 rows (CHECKPOINT=5000). A log printing input 0
therefore does not establish that only one sample was evaluated.
Actual coverage is checked from the source-bound result matrix.

This is an independent, read-only NUMERICAL observation. It cannot
authenticate a Vitis invocation merely from the generated log file.
"""
from __future__ import annotations

from pathlib import Path
import json

from fpgai_research.experiments.hls4ml_csim_numeric import export_csim_numeric


def discover_selected_csim_results(stage_dir) -> Path:
    """Find the unique native output within the original CSim-only clone.

    Do not choose an arbitrary match when multiple solution directories or
    stale runs are present. Never accept symlinks or an external result.
    """
    stage = Path(stage_dir)
    project = stage / "project"
    if (stage.is_symlink() or not stage.is_dir()
            or project.is_symlink() or not project.is_dir()):
        raise ValueError("Expected an ordinary source-bound CSim stage")
    matches = []
    for candidate in project.rglob("csim_results.log"):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("Linked native CSim output is not trusted")
        try:
            rel = candidate.resolve(strict=True).relative_to(project.resolve(strict=True))
        except ValueError as exc:
            raise ValueError("CSim result escaped private project") from exc
        if "tb_data" not in rel.parts:
            continue
        if candidate.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("CSim result file exceeds bounded limit")
        matches.append(candidate)
        if len(matches) > 32:
            raise ValueError("Too many candidate CSim result files")
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one CSim result in the cloned project; "
            f"found {len(matches)}. Provide an explicit --results path if needed.")
    return matches[0]


def audit_native_csim_outputs(stage_dir, output, *, results_file=None, absolute_limit=0.05) -> dict:
    if results_file is None:
        results_file = discover_selected_csim_results(stage_dir)
    audit = export_csim_numeric(
        stage_dir, results_file, output, absolute_limit=absolute_limit)
    metrics = audit["measurements"]
    # Keep all outputs downstream source-bound; don't upgrade receipt to a
    # verified licensed native invocation simply because 16 rows exist.
    return {
        "schema": audit["schema"],
        "model_kind": audit["model_kind"],
        "observed_output_rows": metrics["selected_samples"],
        "selected_top1_disagreements": metrics["selected_top1_disagreements"],
        "max_absolute_probability_difference":
            metrics["selected_probability_max_absolute_difference"],
        "selected_numeric_policy_passed": metrics["selected_numeric_policy_passed"],
        "qualified_vitis_csim_e2": False,
        "native_tool_invocation_independently_attested": False,
        "external_qor_comparison_eligible": False,
    }
