"""Compare two separately qualified hls4ml E1 projects without inferring QoR.

Both candidate projects must derive from the exact same original trained
ONNX/official full-E0 receipt, target, clock and precision. Name-level
tuning must use a real configurable ReuseFactor and source-diff evidence.
A source difference alone never proves achieved II, frequency or resource use.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_e1_reaudit import (
    collect_e1_source_reaudit, _json,
)

SCHEMA = "fpgai-research/hls4ml-matched-e1-pair/v1"


def collect_matched_e1_pair(repository, candidate_dir, full_e0_dir,
                            model_kind, default_project, tuned_project) -> dict:
    first_dir, second_dir = Path(default_project), Path(tuned_project)
    if first_dir.resolve() == second_dir.resolve():
        raise ValueError("Independent E1 baseline/tuned project paths must differ")
    first = collect_e1_source_reaudit(
        repository, candidate_dir, full_e0_dir, model_kind, first_dir)
    second = collect_e1_source_reaudit(
        repository, candidate_dir, full_e0_dir, model_kind, second_dir)
    b = _json(first_dir / "research_baseline.json")
    t = _json(second_dir / "research_baseline.json")
    constant_fields = (
        "schema", "evidence", "model_sha256", "hls4ml_version",
        "onnx_version", "part", "clock_ns", "precision", "backend", "io_type",
        "derived_candidate_full_e0_lineage",
    )
    if any(b.get(k) != t.get(k) for k in constant_fields):
        raise ValueError("E1 comparison target, precision, tool or source lineage differs")
    if (b.get("configuration_granularity") != "model"
            or t.get("configuration_granularity") != "name"):
        raise ValueError("Matched E1 baseline must be model-wide and tuned variant per-layer-name")
    selected = t.get("requested_layer_reuse")
    if (not isinstance(selected, dict) or not selected
            or any(not isinstance(key, str) or not key
                   or type(value) is not int or not 1 <= value <= 4096
                   for key, value in selected.items())):
        raise ValueError("Tuned E1 must declare at least one valid explicit layer control")
    tuned = t.get("hls_config")
    if (not isinstance(tuned, dict) or not isinstance(tuned.get("LayerName"), dict)
            or any(tuned["LayerName"].get(name, {}).get("ReuseFactor") != factor
                   for name, factor in selected.items())):
        raise ValueError("Tuned E1 receipt did not materialize requested controls")
    if (b.get("requested_layer_reuse") != {}
            or first.get("candidate_onnx_sha256") != second.get("candidate_onnx_sha256")
            or first.get("full_e0_receipt_sha256") != second.get("full_e0_receipt_sha256")):
        raise ValueError("Matched E1 candidate source/E0 or default controls changed")
    left = first["generated_cpp_sha256"]
    right = second["generated_cpp_sha256"]
    files = sorted(set(left) | set(right))
    changed = [name for name in files if left.get(name) != right.get(name)]
    same = [name for name in files if left.get(name) == right.get(name)]
    return {
        "schema": SCHEMA,
        "evidence": "E1-source-only-matched-hls4ml-default-vs-name-tuned",
        "model_kind": model_kind,
        "hls4ml_version": b["hls4ml_version"],
        "model_sha256": b["model_sha256"],
        "full_e0_receipt_sha256": first["full_e0_receipt_sha256"],
        "part": b["part"],
        "clock_ns": b["clock_ns"],
        "precision": b["precision"],
        "default_model_reuse_factor": b["reuse_factor"],
        "tuned_model_reuse_factor": t["reuse_factor"],
        "tuned_requested_layer_reuse": dict(sorted(selected.items())),
        "default_cpp_file_count": len(left),
        "tuned_cpp_file_count": len(right),
        "changed_source_paths": changed,
        "unchanged_source_paths": same,
        "different_generated_cpp_source_count": len(changed),
        "source_effect_observed": bool(changed),
        "achieved_hardware_reuse_verified": False,
        "fixed_point_e2_equivalence_verified": False,
        "native_e3_hls_synthesis_verified": False,
        "native_e4_physical_measured": False,
        "fpgai_vs_hls4ml_qor_comparison_eligible": False,
        "vendor_tools_executed": False,
    }


def export_matched_e1_pair(repository, candidate_dir, full_e0_dir, model_kind,
                            default_project, tuned_project, output) -> dict:
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Immutable paired E1 comparison already exists")
    record = collect_matched_e1_pair(
        repository, candidate_dir, full_e0_dir, model_kind,
        default_project, tuned_project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-e1-pair-", dir=destination.parent) as temp:
        stage = Path(temp) / "e1_pair"
        stage.mkdir()
        (stage / "pair_private.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        # No private file paths, ONNX hashes, layer identities, or tensors.
        (stage / "summary.md").write_text(
            "# hls4ml matched default versus per-layer E1 source\n\n"
            f"Model: {model_kind.upper()}; default C++ files: "
            f"{record['default_cpp_file_count']}; tuned C++ files: "
            f"{record['tuned_cpp_file_count']}.\n\n"
            f"Changed C++ paths: {record['different_generated_cpp_source_count']}.\n\n"
            "**A changed source file proves an E1 textual effect only. "
            "An unchanged file does not prove identical architecture. "
            "Neither result establishes realized reuse, fixed-point E2, "
            "synthesized E3, routed E4 or FPGAI superiority.**\n",
            encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Paired E1 comparison output appeared during publication")
        stage.rename(destination)
    return record
