"""R4e: guarded R4d fixed-point-format native Vitis E2/E3 study.

READ-ONLY audit by default; no Vivado or vendor use without both native
opt-in flags. Both source-population hashes and original baseline source
attestation are verified before execution. Never overwrite a prior/partial
native arm. E2 failure or missing effective timing means no E3 resource claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments.cst_r4b_row_retention_native import (
    ARMS, RESOURCES, audit_prepared as audit_r4a,
)
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/R4e-attested-numeric-format-native-E2-E3/v1"
_INPUTS = ("model.cpp", "tb.cpp", "run_hls.tcl", "manifest.json")
_ATTEMPT_PATHS = ("vitis_invocation.json", "logs", "fpgai_hls_project")


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4e missing/symlinked source evidence: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("R4e expected JSON object at " + str(path))
    return data


def audit_prepared(*, study: str | Path, original_study: str | Path) -> dict:
    """Verify exact E1 inputs, full inventory and original R4a baseline."""
    root = Path(study)
    original_root = Path(original_study)
    if (root.is_symlink() or not root.is_dir()
            or original_root.is_symlink() or not original_root.is_dir()
            or root.resolve() == original_root.resolve()
            or original_root.resolve() in root.resolve().parents):
        raise ValueError("R4e original/new source study roots unsafe")
    original_audit = audit_r4a(study=original_root)
    original = _json(original_root / "summary.json")
    original_case = original["rows"][r4d.CASE_INDEX]
    source = _json(root / "summary.json") if _sha(root / "summary.json") else None
    inventory = _json(root / "artifact_sha256.json") if _sha(root / "artifact_sha256.json") else None
    expected = r4d.plan()
    if (source.get("schema") != r4d.SCHEMA
            or source.get("configuration") != expected
            or source.get("original_R4a_case") != original_case["case"]
            or source.get("original_R4a_source_revision") != original_audit["source_revision"]
            or source.get("original_R4a_16_6_E2_result")
                != "FAILED_BOTH_ARMS_RETAINED"
            or source.get("completed_original_GCC_pairs") != len(r4d.FORMATS)
            or source.get("new_GCC_executions") != 2 * len(r4d.FORMATS)
            or source.get("new_native_Vitis_calls") != 0
            or source.get("new_native_Vivado_calls") != 0):
        raise ValueError("R4e R4d source experiment contract mismatched")
    revision = source.get("source_revision")
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("R4e source revision unavailable or invalid")
    if (not isinstance(inventory, dict) or len(inventory) < 10
            or "summary.json" not in inventory):
        raise ValueError("R4e source SHA inventory missing")
    for name, digest in inventory.items():
        if (not isinstance(name, str) or not name or name.startswith("/")
                or "\\" in name or ".." in Path(name).parts
                or name == "artifact_sha256.json"
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("R4e invalid SHA inventory path/hash")
        if _sha(root / name) != digest:
            raise ValueError("R4e source SHA inventory mismatch: " + name)
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != len(r4d.FORMATS):
        raise ValueError("R4e incomplete planned numerical format cohort")
    original_input = None
    original_functional = None
    control_tcl = None
    for idx, (bits, integer) in enumerate(r4d.FORMATS):
        entry = rows[idx]
        if not isinstance(entry, dict) or entry.get("format") != [bits, integer]:
            raise ValueError("R4e numerical format order/count changed")
        folder = root / f"format_{bits:02d}_{integer:02d}"
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R4e expected source format directory absent")
        pair_hashes = {}
        for arm in ARMS:
            row = entry.get(arm)
            if (not isinstance(row, dict)
                    or row.get("E1_GCC_passed") is not True
                    or row.get("Vitis_E2_passed") is not None
                    or row.get("HLS_E3_qualified") is not None):
                raise ValueError("R4e E1 host or unmeasured native gate changed")
            arm_root = folder / arm
            pkg = arm_root / "vitis_source"
            if any(p.is_symlink() for p in (arm_root, pkg)):
                raise ValueError("R4e symlinked source package")
            for file in _INPUTS:
                _sha(pkg / file)
            expected_pkg = row.get("HLS_package_sha256")
            if not isinstance(expected_pkg, dict) or set(expected_pkg) != {
                "model.cpp", "tb.cpp", "run_hls.tcl"
            }:
                raise ValueError("R4e HLS input hash population invalid")
            for file in ("model.cpp", "tb.cpp", "run_hls.tcl"):
                if _sha(pkg / file) != expected_pkg[file]:
                    raise ValueError("R4e prepared native input SHA drift")
            source_sha = row.get("E1_source_cpp_sha256")
            if (source_sha != expected_pkg["model.cpp"]
                    or _sha(arm_root / "source.cpp") != source_sha):
                raise ValueError("R4e actual E1 GCC input differs from Vitis")
            if _sha(arm_root / "host_receipt.json") != row.get(
                    "E1_host_receipt_sha256"):
                raise ValueError("R4e original GCC receipt mismatch")
            host = _json(arm_root / "host_receipt.json")
            if (host.get("source_cpp_sha256") != source_sha
                    or host.get("original_GCC_executions") != 1
                    or host.get("original_Vitis_calls") != 0):
                raise ValueError("R4e source-pinned real GCC verification absent")
            if (original_input is not None
                    and host.get("runtime_input_sha256") != original_input):
                raise ValueError("R4e sampled inputs differ between formats")
            original_input = host.get("runtime_input_sha256")
            meta = _json(pkg / "manifest.json")
            if (meta.get("cpp_sha256") != source_sha
                    or meta.get("precision_bits") != bits
                    or meta.get("integer_bits") != integer
                    or meta.get("part") != expected["part"]
                    or meta.get("clock_ns") != expected["clock_ns"]
                    or meta.get("explicit_clock_uncertainty_ns")
                        != expected["uncertainty_ns"]
                    or meta.get("numerical_tolerance")
                        != expected["absolute_tolerance"]
                    or meta.get("csim_sample_count") != 1
                    or meta.get("csim_outputs_per_sample") != 128
                    or meta.get("testbench_sha256") != expected_pkg["tb.cpp"]):
                raise ValueError("R4e numerical/target/sample contract changed")
            if original_functional is not None and (
                    meta.get("functional_sha256") != original_functional):
                raise ValueError("R4e Functional Model IR/weights changed")
            original_functional = meta.get("functional_sha256")
            if control_tcl is not None and (
                    expected_pkg["run_hls.tcl"] != control_tcl):
                raise ValueError("R4e Vitis target/Tcl differs between formats")
            control_tcl = expected_pkg["run_hls.tcl"]
            pair_hashes[arm] = expected_pkg
            if (bits, integer) == (16, 6) and (
                    source_sha != original_case[arm]["source_sha256"]
                    or expected_pkg
                        != original_case[arm]["source_vitis_package_sha256"]):
                raise ValueError("R4e new 16/6 baseline does not preserve original")
        if pair_hashes["full_H"]["tb.cpp"] != pair_hashes["row_H"]["tb.cpp"]:
            raise ValueError("R4e mismatched CSim reference/testbench for a pair")
    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "original_R4a_source_revision": original_audit["source_revision"],
        "format_pairs": len(rows),
        "source_attested_arms": 2 * len(rows),
        "source_and_testbench_attested": True,
        "original_failed_E2_result_retained": True,
        "original_vendor_calls_during_audit": 0,
    }


def _native_observation(pkg: Path, *, name: str, bits: int, integer: int,
                        arm: str, source_sha: str) -> dict:
    measured = summarize_case(str(pkg))
    qualified = qualify_case(pkg, {
        "case": name + "_" + arm,
        "graph": name,
        "mapping": "original_case_003_mixed",
        "source_storage": arm,
        "source_cpp_sha256": source_sha,
    }, summary=measured)
    passed_e2 = qualified["native_e2_accepted"] is True
    passed_e3 = qualified["native_e3_qualified"] is True
    receipt = measured.get("csim_receipt") or {}
    return {
        "format": [bits, integer],
        "arm": arm,
        "native_invocation_file_present": (pkg / "vitis_invocation.json").is_file(),
        "E2_qualified": passed_e2,
        "E3_qualified": passed_e3,
        "status": qualified["status"],
        "CSim_numerical_receipt_attested": bool(receipt),
        "CSim_failed_outputs": receipt.get("failures") if receipt else None,
        "CSim_max_absolute_error": receipt.get("max_absolute_error") if receipt else None,
        "CSim_rmse": receipt.get("rmse") if receipt else None,
        "CSim_cosine": receipt.get("cosine") if receipt else None,
        "original_source_sha256": source_sha,
        "HLS_estimated_clock_ns": (
            qualified["hls_estimated_clock_ns"] if passed_e3 else None),
        "HLS_worst_cycles": qualified["hls_worst_cycles"] if passed_e3 else None,
        "HLS_CSynth_report_sha256": (
            qualified["hls_report_sha256"] if passed_e3 else None),
        "HLS_estimated_resources": {
            resource: (qualified["estimated_resources"][resource]
                       if passed_e3 else None)
            for resource in RESOURCES
        },
        "physical_E4": None,
    }


def audit_native(*, study: str | Path, original_study: str | Path) -> dict:
    meta = audit_prepared(study=study, original_study=original_study)
    root = Path(study)
    source = _json(root / "summary.json")
    observations, pairs = [], []
    for (bits, integer), entry in zip(r4d.FORMATS, source["rows"]):
        name = f"format_{bits:02d}_{integer:02d}"
        two = []
        for arm in ARMS:
            pkg = root / name / arm / "vitis_source"
            two.append(_native_observation(
                pkg, name=name, bits=bits, integer=integer,
                arm=arm, source_sha=entry[arm]["E1_source_cpp_sha256"]))
        full, row = two
        both_e2 = full["E2_qualified"] and row["E2_qualified"]
        both_e3 = full["E3_qualified"] and row["E3_qualified"]
        pairs.append({
            "format": [bits, integer],
            "both_E2_qualified": both_e2,
            "both_E3_qualified": both_e3,
            "status": ("QUALIFIED_HLS_ESTIMATES_ONLY" if both_e3
                       else "UNKNOWN_OR_REJECTED_HLS_PAIR"),
            "HLS_estimated_resource_delta_row_minus_full": {
                resource: (row["HLS_estimated_resources"][resource]
                           - full["HLS_estimated_resources"][resource])
                if both_e3 and row["HLS_estimated_resources"][resource] is not None
                and full["HLS_estimated_resources"][resource] is not None
                else None
                for resource in RESOURCES
            },
            "HLS_worst_cycles_delta_row_minus_full": (
                row["HLS_worst_cycles"] - full["HLS_worst_cycles"]
                if both_e3 else None),
            "physical_E4_delta": None,
        })
        observations.extend(two)
    return {
        **meta,
        "original_native_invocation_files": sum(
            a["native_invocation_file_present"] for a in observations),
        "E2_qualified_arms": sum(a["E2_qualified"] for a in observations),
        "E3_qualified_arms": sum(a["E3_qualified"] for a in observations),
        "prespecified_matched_pairs": len(pairs),
        "E2_qualified_pairs": sum(p["both_E2_qualified"] for p in pairs),
        "E3_qualified_pairs": sum(p["both_E3_qualified"] for p in pairs),
        "observations": observations,
        "pairs": pairs,
        "native_vendor_calls_during_read_only_audit": 0,
        "physical_E4": None,
    }


def execute(*, study: str | Path, original_study: str | Path,
            start_format: int = 0, max_formats: int = 1,
            execute_native: bool = False,
            confirm_isolated_native: bool = False,
            vitis_executable: str = "vitis_hls",
            timeout_seconds: int = 7200) -> dict:
    meta = audit_prepared(study=study, original_study=original_study)
    count = meta["format_pairs"]
    if (type(start_format) is not int or type(max_formats) is not int
            or start_format < 0 or max_formats < 1
            or start_format + max_formats > count):
        raise ValueError("R4e finite numerical format selection invalid")
    selected = r4d.FORMATS[start_format:start_format + max_formats]
    if not execute_native:
        return {
            **meta, "status": "DRY_RUN_NO_VENDOR_CALLS",
            "selected_formats": [list(f) for f in selected],
            "planned_native_vendor_calls": 2 * len(selected),
            "new_vendor_calls": 0,
        }
    if not confirm_isolated_native:
        raise ValueError("R4e native execution requires explicit confirmation")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
        raise ValueError("R4e bounded native timeout required")
    executable = shutil.which(vitis_executable)
    if not executable:
        raise FileNotFoundError("Vitis HLS executable missing")
    root = Path(study)
    # Prevent partial second runs of the selected population; check ALL
    # selected arms, not merely the first arm, before first native call.
    for bits, integer in selected:
        for arm in ARMS:
            pkg = (root / f"format_{bits:02d}_{integer:02d}"
                   / arm / "vitis_source")
            for item in _ATTEMPT_PATHS:
                path = pkg / item
                if path.exists() or path.is_symlink():
                    raise FileExistsError(
                        "R4e original HLS arm already attempted: " + str(path))
    outputs = []
    source = _json(root / "summary.json")
    for position, (bits, integer) in enumerate(selected, start=start_format):
        entry = source["rows"][position]
        for arm in ARMS:
            name = f"format_{bits:02d}_{integer:02d}"
            pkg = root / name / arm / "vitis_source"
            vendor = run_vitis(
                pkg, executable=executable, timeout_seconds=timeout_seconds)
            outcome = _native_observation(
                pkg, name=name, bits=bits, integer=integer,
                arm=arm, source_sha=entry[arm]["E1_source_cpp_sha256"])
            outcome["tool_process_completed_successfully"] = (
                vendor.get("completed_successfully") is True)
            outputs.append(outcome)
    return {
        **meta,
        "status": "NATIVE_ATTEMPTS_RECORDED_QUALIFICATION_SEPARATE",
        "native_vendor_calls_made": len(outputs),
        "observations": outputs,
        "physical_E4": None,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-study", required=True)
    parser.add_argument("--audit-native", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--confirm-isolated-native", action="store_true")
    parser.add_argument("--start-format", type=int, default=0)
    parser.add_argument("--max-formats", type=int, default=1)
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    report = (audit_native(
        study=args.study, original_study=args.original_study)
        if args.audit_native else execute(
            study=args.study, original_study=args.original_study,
            start_format=args.start_format, max_formats=args.max_formats,
            execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
