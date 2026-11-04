"""Read-only numeric audit of selected hls4ml CSim output, no tool execution.

A numerical result file is not proof of vendor CSim execution. This
records conditional 16-sample agreement and source fingerprints without
upgrading to E2-qualified or cross-framework FPGA hardware evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from fpgai_research.experiments.hls4ml_csim_stage import CSIM_ONLY, SCHEMA as STAGE_SCHEMA


SCHEMA = "fpgai-research/hls4ml-selected-csim-numeric-observation/v1"


def numeric_matrix(text: str, rows=16, columns=10) -> np.ndarray:
    if not isinstance(text, str) or len(text) > 4 * 1024 * 1024:
        raise ValueError("Invalid or oversized private numeric results")
    lines = text.strip().splitlines()
    if len(lines) != rows:
        raise ValueError("Unexpected result row count")
    parsed = []
    for line in lines:
        parts = line.split()
        if len(parts) != columns:
            raise ValueError("Unexpected output feature count")
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError("Non-numeric CSim output") from exc
        if not all(np.isfinite(x) for x in values):
            raise ValueError("Nonfinite CSim output")
        parsed.append(values)
    return np.asarray(parsed, dtype=np.float64)


def compare_numeric_outputs(reference: np.ndarray, result: np.ndarray,
                            *, absolute_limit: float = 0.05) -> dict:
    if (type(absolute_limit) not in (float, int)
            or not np.isfinite(absolute_limit) or not 0 < absolute_limit <= 1):
        raise ValueError("Selected-sample absolute error limit must be in (0, 1]")
    if (reference.shape != (16, 10) or result.shape != (16, 10)
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(result))):
        raise ValueError("Invalid 16-image/10-class numerical arrays")
    diff = np.abs(reference - result)
    disagreements = int(np.sum(np.argmax(reference, axis=1) != np.argmax(result, axis=1)))
    maximum = float(np.max(diff))
    mean = float(np.mean(diff))
    accepted = disagreements == 0 and maximum <= absolute_limit
    return {
        "selected_samples": 16,
        "classes": 10,
        "selected_top1_disagreements": disagreements,
        "selected_probability_max_absolute_difference": maximum,
        "selected_probability_mean_absolute_difference": mean,
        "selected_policy_max_absolute_difference": float(absolute_limit),
        "selected_numeric_policy_passed": accepted,
        "source_vs_csim_selected_numerical_observation_only": True,
        "qualified_vitis_csim_e2": False,
        "full_10000_image_fpga_accuracy_measured": False,
    }


def collect_csim_numeric(stage_dir, results_file, *, absolute_limit=0.05) -> dict:
    root = Path(stage_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing regular CSim-only stage")
    saved = root / "stage.json"
    if saved.is_symlink() or not saved.is_file() or saved.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Missing safe private CSim stage receipt")
    stage = json.loads(saved.read_text(encoding="utf-8"))
    if (not isinstance(stage, dict)
            or stage.get("schema") != STAGE_SCHEMA
            or stage.get("evidence") != "E2-CSIM-STAGED-NOT-EXECUTED"
            or stage.get("build_options") != CSIM_ONLY
            or stage.get("cloned_generated_cpp_unmodified") is not True
            or stage.get("csim_executed") is not False
            or stage.get("external_qor_comparison_eligible") is not False):
        raise ValueError("Not a qualified source-only CSim staging receipt")
    project = root / "project"
    if project.is_symlink() or not project.is_dir():
        raise ValueError("Missing staged original E1 project")
    for filename, digest in stage["generated_cpp_sha256"].items():
        rel = Path(filename)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe generated C++ file path")
        f = project / rel
        if f.is_symlink() or not f.is_file() or sha256(f.read_bytes()).hexdigest() != digest:
            raise ValueError("Generated source changed since E1 staging")
    options = project / "build_opt.tcl"
    if options.is_symlink() or not options.is_file():
        raise ValueError("Unsafe staged tool option file")
    from fpgai_research.experiments.hls4ml_csim_stage import csim_only_options
    text = options.read_text(encoding="utf-8")
    if text != csim_only_options(text):
        raise ValueError("CSim option set changed since staging")
    expected = project / "tb_data" / "tb_output_predictions.dat"
    inputs = project / "tb_data" / "tb_input_features.dat"
    for path, key in [(expected, "tb_reference_sha256"), (inputs, "tb_input_sha256")]:
        if (path.is_symlink() or not path.is_file()
                or sha256(path.read_bytes()).hexdigest() != stage[key]):
            raise ValueError("Staged ONNX reference or input vectors mutated")
    result_file = Path(results_file)
    if result_file.is_symlink() or not result_file.is_file():
        raise ValueError("Missing regular results file")
    # Require the file to reside in the private CSim clone, not arbitrary
    # user-provided data outside the source-bound native build workspace.
    try:
        result_file.resolve(strict=True).relative_to(project.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("Results must be inside the staged CSim project") from exc
    raw = result_file.read_bytes()
    reference = numeric_matrix(expected.read_text(encoding="utf-8"))
    observations = numeric_matrix(raw.decode("utf-8"))
    measures = compare_numeric_outputs(
        reference, observations, absolute_limit=absolute_limit)
    return {
        "schema": SCHEMA,
        "model_kind": stage["model_kind"],
        "e1_project_receipt_sha256": stage["e1_project_receipt_sha256"],
        "candidate_onnx_sha256": stage["candidate_onnx_sha256"],
        "selected_inputs_sha256": stage["selected_inputs_sha256"],
        "csim_result_file_sha256": sha256(raw).hexdigest(),
        "measurements": measures,
        "hls4ml_source_files_unchanged": True,
        "tool_invocation_and_vendor_version_independently_attested": False,
        "qualified_vitis_csim_e2": False,
        "native_e3_e4_executed_by_this_audit": False,
        "external_qor_comparison_eligible": False,
    }


def export_csim_numeric(stage_dir, results_file, output, *, absolute_limit=0.05) -> dict:
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable CSim numeric observation already exists")
    report = collect_csim_numeric(stage_dir, results_file, absolute_limit=absolute_limit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-csim-observation-", dir=dest.parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        (folder / "observation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        metrics = report["measurements"]
        (folder / "summary.md").write_text(
            "# hls4ml selected-image numeric observation, not qualified E2\n\n"
            f'Model: {report["model_kind"].upper()}; samples: 16; '
            f'top-1 disagreements: {metrics["selected_top1_disagreements"]}; '
            f'max absolute difference: {metrics["selected_probability_max_absolute_difference"]:.8g}; '
            f'mean absolute difference: {metrics["selected_probability_mean_absolute_difference"]:.8g}.\n\n'
            "This reads a results file but does not independently verify a real "
            "licensed Vitis CSim invocation, tool version or fixed-point execution. "
            "No E2/E3/E4 qualification or external QoR claim is granted.\n",
            encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Numeric observation appeared during write")
        folder.rename(dest)
    return report
