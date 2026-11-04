"""Preflight two staged hls4ml CSim workspaces for numerical workload parity.

A pair of staged experiments does not establish CSim execution, fixed-point
equivalence, resource usage, or comparative FPGA QoR.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_csim_stage import (
    CSIM_ONLY, SCHEMA as STAGE_SCHEMA, csim_only_options,
)

SCHEMA = "fpgai-research/hls4ml-matched-csim-stage-pair/v1"
MUST_EQUAL = (
    "model_kind", "hls4ml_version", "original_onnx_sha256",
    "candidate_onnx_sha256", "full_e0_receipt_sha256",
    "selected_inputs_sha256", "selected_labels_sha256",
    "tb_input_sha256", "tb_reference_sha256", "test_samples",
    "build_options",
)


def _read_stage(path):
    root = Path(path)
    receipt = root / "stage.json"
    if (root.is_symlink() or not root.is_dir()
            or receipt.is_symlink() or not receipt.is_file()
            or receipt.stat().st_size > 4 * 1024 * 1024):
        raise ValueError("Unsafe or missing private CSim workspace")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != STAGE_SCHEMA
            or record.get("build_options") != CSIM_ONLY
            or record.get("csim_executed") is not False
            or record.get("external_qor_comparison_eligible") is not False):
        raise ValueError("CSim workspace receipt not source-only qualified")
    project = root / "project"
    if project.is_symlink() or not project.is_dir():
        raise ValueError("Missing private CSim clone")
    for filename, sha in [
        ("tb_data/tb_input_features.dat", record["tb_input_sha256"]),
        ("tb_data/tb_output_predictions.dat", record["tb_reference_sha256"]),
    ]:
        f = project / filename
        if f.is_symlink() or not f.is_file() or sha256(f.read_bytes()).hexdigest() != sha:
            raise ValueError("Numerical input/reference changed after staging")
    for rel, sha in record["generated_cpp_sha256"].items():
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe generated C++ source identity")
        cpp = project / path
        if cpp.is_symlink() or not cpp.is_file() or sha256(cpp.read_bytes()).hexdigest() != sha:
            raise ValueError("Cloned E1 C++ source differs from stage receipt")
    options = project / "build_opt.tcl"
    if options.is_symlink() or not options.is_file():
        raise ValueError("Missing or linked CSim-only options")
    text = options.read_text(encoding="utf-8")
    if text != csim_only_options(text):
        raise ValueError("CSim-only option file has changed")
    return record


def collect_csim_stage_pair(default_stage, tuned_stage):
    if Path(default_stage).resolve() == Path(tuned_stage).resolve():
        raise ValueError("CSim-only experiments must be independent directories")
    default = _read_stage(default_stage)
    tuned = _read_stage(tuned_stage)
    if any(default.get(key) != tuned.get(key) for key in MUST_EQUAL):
        raise ValueError("Default and tuned CSim inputs, source or settings disagree")
    if default["test_samples"] != 16:
        raise ValueError("Only 16 attested selected MNIST samples supported")
    files = sorted(set(default["generated_cpp_sha256"]) | set(tuned["generated_cpp_sha256"]))
    changed = [
        f for f in files if
        default["generated_cpp_sha256"].get(f) != tuned["generated_cpp_sha256"].get(f)
    ]
    return {
        "schema": SCHEMA,
        "model_kind": default["model_kind"],
        "hls4ml_version": default["hls4ml_version"],
        "shared_selected_inputs_sha256": default["tb_input_sha256"],
        "shared_selected_reference_sha256": default["tb_reference_sha256"],
        "selected_samples": 16,
        "matched_selected_testbench_data": True,
        "both_csim_only": True,
        "e1_source_difference_count": len(changed),
        "e1_source_effect_observed": bool(changed),
        "csim_executed_by_this_check": False,
        "e2_numerical_qualified": False,
        "native_e3_e4_executed": False,
        "cross_framework_qor_comparison_eligible": False,
    }


def export_csim_stage_pair(default_stage, tuned_stage, output):
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable paired CSim preflight already exists")
    record = collect_csim_stage_pair(default_stage, tuned_stage)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-csim-pair-", dir=dest.parent) as tmp:
        folder = Path(tmp) / "pair"
        folder.mkdir()
        (folder / "pair_private.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (folder / "summary.md").write_text(
            "# Paired hls4ml CSim staging preflight\n\n"
            f'Model: {record["model_kind"].upper()}; selected images: 16. '
            "Default and tuned testbench inputs/references are byte-identical; "
            "both stage copies use CSim-only Vitis options.\n\n"
            "No CSim or synthesis ran, and no fixed-point E2, E3, E4, "
            "hardware benefit or FPGAI/hls4ml QoR comparison was established.\n")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Paired CSim output appeared concurrently")
        folder.rename(dest)
    return record
