"""R4l: guarded new Vitis E2/E3 on R4k schedule-matched source layouts.

Default is SHA-attested, vendor-FREE audit/plan. Requires BOTH explicit
native opt-in flags. Vendor attempts are CREATE-ONCE within the NEW R4k
source-only evidence, never any original R4a/R4d vendor case.
E2 numerical qualification gates E3 resource/timing interpretation.
Source events are matched; HLS hardware schedule is NOT assumed matched.
No Vivado or physical E4 observation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from fpgai_research.experiments import cst_r4k_matched_row_schedule_storage as r4k
from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/R4l-isolated-same-source-schedule-native-E2-E3/v1"
ATTEMPTS = ("vitis_invocation.json", "logs", "fpgai_hls_project")
SOURCES = ("model.cpp", "tb.cpp", "run_hls.tcl", "manifest.json")
_RESOURCES = ("bram_18k", "dsp", "lut", "ff", "uram")
_SOURCE_TAG = re.compile(r"[0-9a-f]{40}\Z")
_SHA_TAG = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4l original/new attested file missing or symlinked: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("R4l JSON is not an object: " + str(path))
    return data


def audit_prepared(*, study: str | Path, original_r4d: str | Path,
                   original_r4a: str | Path) -> dict:
    """Recheck all create-only E1 source package hashes before ANY vendor use."""
    root, r4d, r4a = Path(study), Path(original_r4d), Path(original_r4a)
    if (root.is_symlink() or not root.is_dir()
            or r4d.is_symlink() or r4a.is_symlink()
            or not r4d.is_dir() or not r4a.is_dir()
            or len({root.resolve(), r4d.resolve(), r4a.resolve()}) != 3
            or any(root.resolve() == r.resolve() or r.resolve() in root.resolve().parents
                   or root.resolve() in r.resolve().parents
                   for r in (r4d, r4a))):
        raise ValueError("R4l source/original evidence roots unsafe")
    config = r4k.plan()
    old = r4k._read_only_original_preflight(r4d, r4a, config)
    report = _json(root / "summary.json")
    inventory = _json(root / "artifact_sha256.json")
    if (report.get("schema") != r4k.SCHEMA
            or report.get("configuration") != config
            or report.get("source_only_E1_verified") is not True
            or report.get("original_preflight") != old
            or report.get("HLS_E2_run") is not False
            or report.get("HLS_E3_run") is not False
            or report.get("physical_E4_run") is not False
            or report.get("new_vendor_Vitis_calls") != 0
            or report.get("new_vendor_Vivado_calls") != 0
            or not isinstance(report.get("source_revision"), str)
            or not _SOURCE_TAG.fullmatch(report["source_revision"])):
        raise ValueError("R4l original create-only R4k E1 source contract drift")
    e1 = report.get("E1")
    if (not isinstance(e1, dict)
            or e1.get("status") != "MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED"
            or e1.get("original_host_gcc_compilations") != 2
            or e1.get("original_host_gcc_executions") != 2
            or e1.get("new_native_Vitis_calls") != 0
            or e1.get("all_intermediates_numerically_equal") is not True
            or e1.get("all_final_outputs_numerically_equal") is not True
            or e1.get("source_schedule_sha256") != config["source_schedule_sha256"]
            or e1.get("source_scratch_elements_saved") != 240):
        raise ValueError("R4l original executed E1 qualification missing")
    expected_files = {"summary.json"}
    for arm in r4k.ARMS:
        expected_files.update((
            f"{arm}.host_stdout.txt",
            f"{arm}/source.cpp",
            f"{arm}/manifest.json",
            *(f"{arm}/vitis_source/{name}" for name in SOURCES),
        ))
    if (not isinstance(inventory, dict) or
            not expected_files.issubset(inventory) or
            any(not isinstance(name, str) or name.startswith("/")
                or "\\" in name or ".." in Path(name).parts
                or name == "artifact_sha256.json"
                or not isinstance(value, str) or not _SHA_TAG.fullmatch(value)
                for name, value in inventory.items())):
        raise ValueError("R4l source artifact SHA inventory incomplete/unsafe")
    for relpath, digest in sorted(inventory.items()):
        if _sha(root / relpath) != digest:
            raise ValueError("R4l R4k create-only original SHA inventory mismatch: " + relpath)
    for arm in r4k.ARMS:
        trace_path = root / f"{arm}.host_stdout.txt"
        trace = trace_path.read_text()
        trace_without_addresses = "\n".join(
            line for line in trace.splitlines()
            if not line.startswith("FPGAI_SLOT ")
        ) + "\n"
        if sha256(trace_without_addresses.encode()).hexdigest() != (
                e1.get("source_logic_trace_without_addresses_sha256")):
            raise ValueError("R4l original E1 host issue/value trace changed")
    package_hashes = {}
    target = None
    for arm in r4k.ARMS:
        arm_root = root / arm
        pkg = arm_root / "vitis_source"
        if (arm_root.is_symlink() or pkg.is_symlink()
                or not arm_root.is_dir() or not pkg.is_dir()):
            raise ValueError("R4l original arm source directory missing/symlinked")
        m = _json(pkg / "manifest.json")
        for path in SOURCES:
            _sha(pkg / path)
        model_hash = _sha(pkg / "model.cpp")
        if (model_hash != _sha(arm_root / "source.cpp")
                or model_hash != e1["full_source_sha256"
                                     if arm == "full_storage"
                                     else "row_source_sha256"]
                or model_hash != config["source_full_storage_sha256"
                                        if arm == "full_storage"
                                        else "source_row_original_sha256"]
                or m.get("cpp_sha256") != model_hash
                or m.get("precision_bits") != 32
                or m.get("integer_bits") != 6
                or m.get("part") != config["part"]
                or m.get("clock_ns") != config["clock_ns"]
                or m.get("explicit_clock_uncertainty_ns") != config["uncertainty_ns"]
                or m.get("numerical_tolerance") != config["csim_abs_tolerance"]
                or m.get("csim_sample_count") != 1
                or m.get("csim_outputs_per_sample") != 128
                or m.get("testbench_sha256") != _sha(pkg / "tb.cpp")
                or _sha(pkg / "run_hls.tcl") != old["original_tcl_sha256"]):
            raise ValueError("R4l input source/Tcl/numeric contract changed")
        current = {
            "model.cpp": model_hash,
            "tb.cpp": _sha(pkg / "tb.cpp"),
            "run_hls.tcl": _sha(pkg / "run_hls.tcl"),
            "functional_sha256": m.get("functional_sha256"),
            "architecture_sha256": m.get("architecture_sha256"),
        }
        if not all(isinstance(v, str) and _SHA_TAG.fullmatch(v)
                   for v in (current["functional_sha256"],
                             current["architecture_sha256"])):
            raise ValueError("R4l missing Functional Model and architecture identities")
        if target is not None and (current["tb.cpp"] != target["tb.cpp"]
                 or current["run_hls.tcl"] != target["run_hls.tcl"]
                 or current["functional_sha256"] != target["functional_sha256"]
                 or current["architecture_sha256"] != target["architecture_sha256"]):
            raise ValueError("R4l matched HLS packages differ in oracle/target/model")
        target = current
        package_hashes[arm] = current
    return {
        "schema": SCHEMA,
        "source_revision": report["source_revision"],
        "original_R4d_source_revision": old["source_revision"],
        "original_R4a_source_revision": old["original_R4a_source_revision"],
        "E1_matched_pair_accepted": True,
        "paired_HLS_source_arms": 2,
        "new_source_schedule_sha256": e1["source_schedule_sha256"],
        "source_only_storage_delta_elements": 240,
        "package_hashes": package_hashes,
        "original_vendor_calls_during_audit": 0,
        "physical_E4": None,
    }


def _native_observation(pkg: Path, *, arm: str, source_sha: str) -> dict:
    summary = summarize_case(str(pkg))
    qualified = qualify_case(pkg, {
        "case": "R4l_same_row_source_schedule_" + arm,
        "graph": "R4k_original_case_003",
        "mapping": "R4k_same_row_source_schedule",
        "source_storage": arm,
        "source_cpp_sha256": source_sha,
    }, summary=summary)
    e2 = qualified["native_e2_accepted"] is True
    e3 = qualified["native_e3_qualified"] is True
    csim = summary.get("csim_receipt") or {}
    return {
        "arm": arm,
        "native_invocation_file_present": (pkg / "vitis_invocation.json").is_file(),
        "E2_qualified": e2,
        "E3_qualified": e3,
        "status": qualified["status"],
        "CSim_failed_outputs": csim.get("failures") if csim else None,
        "CSim_max_absolute_error": csim.get("max_absolute_error") if csim else None,
        "CSim_rmse": csim.get("rmse") if csim else None,
        "source_sha256": source_sha,
        "HLS_estimated_clock_ns": (
            qualified["hls_estimated_clock_ns"] if e3 else None),
        "HLS_worst_cycles": qualified["hls_worst_cycles"] if e3 else None,
        "HLS_CSynth_report_sha256": (
            qualified["hls_report_sha256"] if e3 else None),
        "HLS_estimated_resources": {
            key: (qualified["estimated_resources"][key] if e3 else None)
            for key in _RESOURCES
        },
        "physical_E4": None,
    }


def audit_native(*, study: str | Path, original_r4d: str | Path,
                 original_r4a: str | Path) -> dict:
    source = audit_prepared(
        study=study, original_r4d=original_r4d, original_r4a=original_r4a)
    root = Path(study)
    arms = [
        _native_observation(root / arm / "vitis_source", arm=arm,
                            source_sha=source["package_hashes"][arm]["model.cpp"])
        for arm in r4k.ARMS
    ]
    before, after = arms
    both_e2 = before["E2_qualified"] and after["E2_qualified"]
    both_e3 = before["E3_qualified"] and after["E3_qualified"]
    return {
        **source,
        "status": ("PAIRED_QUALIFIED_HLS_ESTIMATES_ONLY"
                   if both_e3 else "UNKNOWN_OR_REJECTED_HLS_PAIR"),
        "new_original_vendor_invocation_files": sum(
            a["native_invocation_file_present"] for a in arms),
        "E2_qualified_arms": sum(a["E2_qualified"] for a in arms),
        "E3_qualified_arms": sum(a["E3_qualified"] for a in arms),
        "both_E2_qualified": both_e2,
        "both_E3_qualified": both_e3,
        "observations": arms,
        "estimated_resource_delta_row_minus_full": {
            key: (
                after["HLS_estimated_resources"][key]
                - before["HLS_estimated_resources"][key]
                if both_e3
                and after["HLS_estimated_resources"][key] is not None
                and before["HLS_estimated_resources"][key] is not None
                else None)
            for key in _RESOURCES
        },
        "HLS_worst_cycles_delta_row_minus_full": (
            after["HLS_worst_cycles"] - before["HLS_worst_cycles"]
            if both_e3 else None),
        "HLS_estimated_clock_delta_ns_row_minus_full": (
            after["HLS_estimated_clock_ns"] - before["HLS_estimated_clock_ns"]
            if both_e3 else None),
        "physical_E4": None,
        "new_vendor_calls_during_read_only_audit": 0,
        "original_sources_modified": False,
    }


def execute(*, study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, execute_native: bool = False,
            confirm_isolated_native: bool = False,
            vitis_executable: str = "vitis_hls",
            timeout_seconds: int = 7200) -> dict:
    meta = audit_prepared(
        study=study, original_r4d=original_r4d, original_r4a=original_r4a)
    if not execute_native:
        return {
            **meta, "status": "DRY_RUN_NO_VENDOR_CALLS",
            "planned_new_vendor_calls": 2,
            "new_vendor_calls": 0,
            "original_sources_modified": False,
        }
    if not confirm_isolated_native:
        raise ValueError("R4l native execution requires explicit confirmation")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
        raise ValueError("R4l bounded Vitis native timeout required")
    exe = shutil.which(vitis_executable)
    if exe is None:
        raise FileNotFoundError("R4l Vitis executable unavailable")
    root = Path(study)
    for arm in r4k.ARMS:
        folder = root / arm / "vitis_source"
        for name in ATTEMPTS:
            p = folder / name
            if p.exists() or p.is_symlink():
                raise FileExistsError(
                    "R4l original matched arm already attempted: " + str(p))
    result = []
    for arm in r4k.ARMS:
        folder = root / arm / "vitis_source"
        result.append({
            **_native_observation(
                folder, arm=arm,
                source_sha=meta["package_hashes"][arm]["model.cpp"]),
            "new_vendor_invocation": run_vitis(
                folder, executable=exe,
                timeout_seconds=timeout_seconds).get("completed_successfully"),
        })
    return {
        **meta,
        "status": "NEW_VENDOR_ATTEMPTS_RECORDED_QUALIFICATION_SEPARATE",
        "original_native_vendor_calls_made": 2,
        "observations": result,
        "original_sources_modified": False,
        "physical_E4": None,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--audit-native", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--confirm-isolated-native", action="store_true")
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    report = (
        audit_native(study=args.study,
                     original_r4d=args.original_r4d,
                     original_r4a=args.original_r4a)
        if args.audit_native
        else execute(
            study=args.study,
            original_r4d=args.original_r4d,
            original_r4a=args.original_r4a,
            execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
