"""R4d: create-only source packages for a matched fixed-point format sweep.

A new numerical investigation of the ORIGINAL R4a case_003 graph/mapping,
not a retry or edit of the E2-rejected Vitis result. All formats share
weights, inputs, 0.02 absolute tolerance, target, clock and mapping.
The existing resolver admits 16/32-bit widths; no unsupported width or
post-hoc numerical relaxation is introduced. This command runs only host
GCC E1, never Vitis or Vivado.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp,
)
from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments.cst_r4b_row_retention_native import (
    ARMS, audit_prepared,
)
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.validation.r2g_fanout_cpp import verify_fanout_gcc
from fpgai_research.validation.row_retained_fanout_cpp import (
    verify_row_retained_gcc,
)

SCHEMA = "fpgai-research/R4d-matched-numeric-format-E1-source-packages/v1"
CASE_INDEX = 3
FORMATS = ((16, 6), (32, 6), (32, 8))


def plan() -> dict:
    graph = r4a.CASES[CASE_INDEX]
    # All proposed values must be legal under the EXISTING resolver. The
    # original R4a 16/6 baseline is retained as a separate new control.
    for width, integer in FORMATS:
        arch = r4a._case(graph, precision_bits=width, integer_bits=integer)
        if {(
            p.effective.precision_bits, p.effective.integer_bits
        ) for p in arch.points} != {(width, integer)}:
            raise ValueError("R4d model-global numerical contract unsupported")
    return {
        "schema": SCHEMA,
        "original_R4a_case": f"case_{CASE_INDEX:03d}",
        "graph": [graph[0], list(graph[1]), graph[2]],
        "predeclared_formats_total_integer_bits": [list(v) for v in FORMATS],
        "matched_storage_pairs": len(FORMATS),
        "source_GCC_arms": 2 * len(FORMATS),
        "original_failure_retained": True,
        "fixed_reference": "float32 FunctionalModel IR",
        "absolute_tolerance": r4a.CSIM_ABS_TOLERANCE,
        "part": r4a.PART,
        "clock_ns": r4a.CLOCK_NS,
        "uncertainty_ns": r4a.UNCERTAINTY_NS,
        "E1_host_float_only": True,
        "native_Vitis_calls": 0,
        "native_Vivado_calls": 0,
    }


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4d original source input absent or symlinked")
    return sha256(path.read_bytes()).hexdigest()


def run(*, original_study: str | Path, output: str | Path) -> dict:
    configuration = plan()
    original_evidence = audit_prepared(study=original_study)
    original_root = Path(original_study)
    source_summary = json.loads((original_root / "summary.json").read_text())
    if (source_summary["configuration"]["profile"] != "paper"
            or len(source_summary["rows"]) <= CASE_INDEX):
        raise ValueError("R4d requires original R4a paper case_003")
    original_case = source_summary["rows"][CASE_INDEX]
    if (original_case["case"] != "case_003"
            or original_case["graph"] != configuration["graph"]):
        raise ValueError("R4d requires original R4a paper case_003")
    for arm in ARMS:
        meta_path = (original_root / "case_003" / arm
                     / "vitis_source" / "manifest.json")
        _digest(meta_path)
        meta = json.loads(meta_path.read_text())
        if (meta.get("precision_bits") != 16
                or meta.get("integer_bits") != 6
                or meta.get("numerical_tolerance")
                    != configuration["absolute_tolerance"]):
            raise ValueError("Original R4a numerical baseline altered")
    root = Path(output)
    if root.resolve() == original_root.resolve() or (
            original_root.resolve() in root.resolve().parents):
        raise ValueError("R4d cannot write inside original evidence folder")
    if root.exists() or root.is_symlink():
        raise FileExistsError("R4d output is create-only")
    # Check predeclared baseline SHA before allocating any new original
    # output. This prevents an accidental model, weight, mapping or emitter
    # drift from making an invalid "same experiment" comparison.
    graph = r4a.CASES[CASE_INDEX]
    baseline_arch = r4a._case(graph, precision_bits=16, integer_bits=6)
    baseline = {
        "full_H": emit_fanout_cpp(baseline_arch),
        "row_H": emit_row_retained_fanout_cpp(baseline_arch),
    }
    for arm in ARMS:
        if baseline[arm].manifest["source_cpp_sha256"] != (
                original_case[arm]["source_sha256"]):
            raise ValueError("R4d original 16/6 generated source SHA changed")
    revision = r4a._git_revision()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        x = r4a._sample(graph[1][0], graph[1][1])
        for width, integer in FORMATS:
            arch = r4a._case(
                graph, precision_bits=width, integer_bits=integer)
            folder = root / f"format_{width:02d}_{integer:02d}"
            folder.mkdir()
            emissions = {
                "full_H": emit_fanout_cpp(arch),
                "row_H": emit_row_retained_fanout_cpp(arch),
            }
            arms = {}
            for arm in ARMS:
                arm_dir = folder / arm
                arm_dir.mkdir()
                emission = emissions[arm]
                (arm_dir / "source.cpp").write_text(emission.cpp)
                trace = arm_dir / "host_stdout.txt"
                verify = (verify_fanout_gcc if arm == "full_H"
                          else verify_row_retained_gcc)
                receipt = verify(arch, emission, runtime_input=x,
                                 original_trace_output=trace)
                (arm_dir / "host_receipt.json").write_text(
                    json.dumps(receipt, sort_keys=True, indent=2) + "\n")
                package_dir = arm_dir / "vitis_source"
                prepared = write_vitis_package(
                    str(package_dir), arch.functional,
                    r4a._vitis_emission(arch, emission), x,
                    part=r4a.PART, clock_ns=r4a.CLOCK_NS,
                    clock_uncertainty_ns=r4a.UNCERTAINTY_NS,
                    tolerance=r4a.CSIM_ABS_TOLERANCE)
                checks = {
                    key: _digest(package_dir / key)
                    for key in ("model.cpp", "tb.cpp", "run_hls.tcl")
                }
                if (checks["model.cpp"] != receipt["source_cpp_sha256"]
                        or checks["tb.cpp"] != prepared["testbench_sha256"]
                        or prepared["precision_bits"] != width
                        or prepared["integer_bits"] != integer):
                    raise ValueError("R4d E1 source/package/numeric SHA mismatch")
                if (checks["run_hls.tcl"]
                        != original_case[arm]["source_vitis_package_sha256"]["run_hls.tcl"]):
                    raise ValueError("R4d original target/Tcl constraints changed")
                if (width, integer) == (16, 6):
                    control = original_case[arm]["source_vitis_package_sha256"]
                    if checks != control:
                        raise ValueError("R4d original fixed-format baseline package changed")
                arms[arm] = {
                    "E1_host_receipt_sha256": _digest(arm_dir / "host_receipt.json"),
                    "E1_source_cpp_sha256": checks["model.cpp"],
                    "HLS_package_sha256": checks,
                    "E1_GCC_passed": True,
                    "Vitis_E2_passed": None,
                    "HLS_E3_qualified": None,
                }
            if (arms["full_H"]["HLS_package_sha256"]["tb.cpp"]
                    != arms["row_H"]["HLS_package_sha256"]["tb.cpp"]):
                raise ValueError("R4d two arms have mismatched numerical oracle")
            rows.append({
                "format": [width, integer],
                "full_H": arms["full_H"],
                "row_H": arms["row_H"],
                "storage_delta_E3": None,
                "physical_delta_E4": None,
            })
        result = {
            "schema": SCHEMA,
            "configuration": configuration,
            "source_revision": revision,
            "original_R4a_source_revision": original_evidence["source_revision"],
            "original_R4a_case": original_case["case"],
            "original_R4a_16_6_E2_result": "FAILED_BOTH_ARMS_RETAINED",
            "new_GCC_executions": 2 * len(rows),
            "completed_original_GCC_pairs": len(rows),
            "new_native_Vitis_calls": 0,
            "new_native_Vivado_calls": 0,
            "new_Vitis_E2_results": None,
            "new_HLS_E3_qualified_pairs": None,
            "new_physical_E4_results": None,
            "rows": rows,
            "limits": [
                "No Vitis CSim, HLS synthesis or FPGA implementation executed",
                "GCC host float cannot attribute native fixed-point error",
                "Format is fixed within each storage pair, not across formats",
                "Original failed 16/6 E2 comparison remains in original cohort",
            ],
        }
        (root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True,
                       allow_nan=False) + "\n")
        artifacts = {
            str(path.relative_to(root)): _digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        }
        (root / "artifact_sha256.json").write_text(
            json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        (root / "FAILED.json").write_text(json.dumps({
            "source_revision": revision,
            "completed_format_pairs": len(rows),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:4000],
        }, indent=2, sort_keys=True) + "\n")
        raise
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-study")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.output is None:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    if not args.original_study:
        parser.error("--original-study is required for create-only output")
    result = run(original_study=args.original_study, output=args.output)
    print(json.dumps({
        "schema": SCHEMA,
        "source_revision": result["source_revision"],
        "GCC_pairs": result["completed_original_GCC_pairs"],
        "native_Vitis_calls": 0,
        "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
