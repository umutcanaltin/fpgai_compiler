"""R4b: guarded original Vitis HLS E2/E3 validation of R4a source pairs.

No native execution without BOTH --execute-native and
--confirm-isolated-native. All original generated source, host E1,
testbench and Tcl hashes are rechecked BEFORE a vendor call. Each
new vendor case is create-once; partial/interrupted vendor directories
are never reused or overwritten. The report includes ALL selected
matched pairs; failed timing or missing results remain UNKNOWN.
Nothing here claims routed physical BRAM or Fmax.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r4a_row_retention_study import (
    SCHEMA as SOURCE_SCHEMA, plan,
)
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/R4b-source-attested-row-retained-native-E2-E3/v1"
RESOURCES = ("bram_18k", "lut", "ff", "dsp", "uram")
ARMS = ("full_H", "row_H")


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4b input evidence absent or symlinked: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def audit_prepared(*, study: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4b original R4a source study unavailable")
    summary_path = root / "summary.json"
    digest_path = root / "artifact_sha256.json"
    _sha(summary_path)
    summary = json.loads(summary_path.read_text())
    _sha(digest_path)
    inventory = json.loads(digest_path.read_text())
    if not isinstance(inventory, dict):
        raise ValueError("R4b original SHA inventory missing")
    for name, expected in inventory.items():
        if (not isinstance(name, str) or name.startswith("/")
                or ".." in Path(name).parts or
                not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("R4b original inventory path/hash invalid")
        if _sha(root / name) != expected:
            raise ValueError("R4b original source artifact SHA mismatch: " + name)
    profile = summary.get("configuration", {}).get("profile")
    expected = plan(profile=profile)
    if (summary.get("schema") != SOURCE_SCHEMA
            or summary.get("configuration") != expected
            or summary.get("native_Vitis_calls") != 0
            or summary.get("native_Vivado_calls") != 0
            or summary.get("original_GCC_compilations")
                != expected["matched_arms"]
            or summary.get("original_GCC_executions")
                != expected["matched_arms"]
            or summary.get("completed_original_GCC_matched_pairs")
                != expected["prespecified_graphs"]):
        raise ValueError("R4b original matched E1 source population invalid")
    revision = summary.get("source_revision")
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise ValueError("R4b original source SHA invalid")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != expected["prespecified_graphs"]:
        raise ValueError("R4b source pair population incomplete")
    for i, row in enumerate(rows):
        cid = f"case_{i:03d}"
        if row.get("case") != cid or row.get("graph") != expected["configs"][i]:
            raise ValueError("R4b prespecified case identity/mapping changed")
        tcl = set()
        tb = set()
        for arm in ARMS:
            r = row.get(arm)
            if not isinstance(r, dict):
                raise ValueError("R4b missing original GCC arm receipt")
            folder = root / cid / arm
            if _sha(folder / "source.cpp") != r.get("source_sha256"):
                raise ValueError("R4b source SHA differs from real GCC input")
            package = folder / "vitis_source"
            if _sha(package / "model.cpp") != r.get("source_sha256"):
                raise ValueError("R4b HLS source differs from E1 original")
            manifest = json.loads((package / "manifest.json").read_text())
            if (manifest.get("cpp_sha256") != r["source_sha256"]
                    or manifest.get("clock_ns") != expected["clock_ns"]
                    or manifest.get("explicit_clock_uncertainty_ns")
                        != expected["clock_uncertainty_ns"]
                    or manifest.get("numerical_tolerance")
                        != expected["csim_absolute_tolerance"]
                    or manifest.get("part") != expected["part"]):
                raise ValueError("R4b HLS package source/target contract changed")
            checksums = r.get("source_vitis_package_sha256") or {}
            for name in ("model.cpp", "tb.cpp", "run_hls.tcl"):
                if _sha(package / name) != checksums.get(name):
                    raise ValueError("R4b package source/testbench/Tcl SHA mismatch")
            if _sha(folder / "host_receipt.json") != r.get(
                    "GCC_host_receipt_sha256"):
                raise ValueError("R4b original E1 GCC receipt SHA mismatch")
            host = json.loads((folder / "host_receipt.json").read_text())
            if (host.get("source_cpp_sha256") != r["source_sha256"]
                    or host.get("original_GCC_executions") != 1
                    or host.get("original_Vitis_calls") != 0):
                raise ValueError("R4b original E1 host verification is missing")
            tcl.add(checksums["run_hls.tcl"])
            tb.add(checksums["tb.cpp"])
        if len(tcl) != 1 or len(tb) != 1:
            raise ValueError("R4b same-network two arms have unmatched HLS controls")
    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "profile": profile,
        "original_complete_E1_pairs": len(rows),
        "original_HLS_source_packages": 2*len(rows),
        "source_and_testbench_attested": True,
        "original_vendor_calls_during_audit": 0,
        "physical_E4": None,
    }


def audit_native(*, study: str | Path) -> dict:
    meta = audit_prepared(study=study)
    root = Path(study)
    source = json.loads((root/"summary.json").read_text())
    cases = []
    pairs = []
    for row in source["rows"]:
        observations = []
        for arm in ARMS:
            pkg = root / row["case"] / arm / "vitis_source"
            record = row[arm]
            expected = {
                "case": f'{row["case"]}_{arm}',
                "graph": str(row["graph"]),
                "mapping": str(row["graph"][2]),
                "source_storage": arm,
                "source_cpp_sha256": record["source_sha256"],
            }
            measured = summarize_case(str(pkg))
            qualified = qualify_case(pkg, expected, summary=measured)
            e3 = qualified["native_e3_qualified"] is True
            observations.append({
                "case": row["case"], "arm": arm,
                "original_vendor_receipt_present":
                    (pkg / "vitis_invocation.json").is_file(),
                "E2_qualified": qualified["native_e2_accepted"] is True,
                "E3_qualified": e3,
                "status": qualified["status"],
                "source_cpp_sha256": record["source_sha256"],
                "worst_cycles": qualified["hls_worst_cycles"] if e3 else None,
                "estimated_clock_ns":
                    qualified["hls_estimated_clock_ns"] if e3 else None,
                "estimated_resources": {
                    name: qualified["estimated_resources"][name] if e3
                    else None for name in RESOURCES
                },
                "HLS_CSynth_report_sha256":
                    qualified["hls_report_sha256"] if e3 else None,
                "physical_BRAM": None,
            })
        a, b = observations
        both = a["E3_qualified"] and b["E3_qualified"]
        pairs.append({
            "case": row["case"], "control": "full_H",
            "intervention": "row_H",
            "both_E2_qualified": a["E2_qualified"] and b["E2_qualified"],
            "both_E3_qualified": both,
            "status": "QUALIFIED_HLS_ESTIMATES_ONLY" if both
                      else "UNKNOWN_OR_E3_REJECTED",
            "HLS_estimated_resource_delta_intervention_minus_control": {
                name: (b["estimated_resources"][name]
                       - a["estimated_resources"][name])
                if both and b["estimated_resources"][name] is not None
                and a["estimated_resources"][name] is not None else None
                for name in RESOURCES
            },
            "HLS_worst_cycles_delta": (
                b["worst_cycles"]-a["worst_cycles"] if both else None),
            "physical_BRAM_delta": None,
            "routed_clock_delta": None,
        })
        cases.extend(observations)
    return {
        **meta,
        "original_native_invocation_receipts": sum(
            r["original_vendor_receipt_present"] for r in cases),
        "original_E2_qualified": sum(r["E2_qualified"] for r in cases),
        "original_E3_qualified": sum(r["E3_qualified"] for r in cases),
        "planned_HLS_pairs": len(pairs),
        "qualified_HLS_pairs": sum(p["both_E3_qualified"] for p in pairs),
        "unqualified_HLS_pairs": sum(not p["both_E3_qualified"] for p in pairs),
        "cases": cases,
        "pairs": pairs,
        "vendor_calls_during_read_only_audit": 0,
        "physical_E4": None,
        "achieved_Fmax": None,
    }


def execute(
    *, study: str | Path, start_case: int = 0, max_cases: int = 1,
    execute_native: bool = False, confirm_isolated_native: bool = False,
    vitis_executable: str = "vitis_hls", timeout_seconds: int = 7200,
) -> dict:
    meta = audit_prepared(study=study)
    root = Path(study)
    source = json.loads((root/"summary.json").read_text())
    count = len(source["rows"])
    if (type(start_case) is not int or type(max_cases) is not int
            or start_case < 0 or max_cases < 1
            or start_case + max_cases > count):
        raise ValueError("R4b finite new source case range invalid")
    selection = source["rows"][start_case:start_case + max_cases]
    if not execute_native:
        return {
            **meta, "status": "DRY_RUN_NO_VENDOR_CALLS",
            "selected_matched_case_ids": [r["case"] for r in selection],
            "planned_vendor_calls": 2*len(selection),
            "new_vendor_calls": 0,
        }
    if not confirm_isolated_native:
        raise ValueError("R4b native execution requires explicit confirmation")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 86400:
        raise ValueError("R4b finite Vitis timeout required")
    exe = shutil.which(vitis_executable)
    if exe is None:
        raise FileNotFoundError("Vitis HLS executable unavailable")
    # Check every original selected arm BEFORE making any native calls.
    for row in selection:
        for arm in ARMS:
            folder = root/row["case"]/arm/"vitis_source"
            for artifact in ("vitis_invocation.json",
                             "logs", "fpgai_hls_project"):
                location = folder/artifact
                if location.exists() or location.is_symlink():
                    raise FileExistsError(
                        "R4b original HLS arm already attempted: " + str(location))
    outcomes = []
    for row in selection:
        for arm in ARMS:
            folder = root/row["case"]/arm/"vitis_source"
            receipt = run_vitis(
                folder, executable=exe, timeout_seconds=timeout_seconds)
            measured = summarize_case(str(folder))
            qualified = qualify_case(folder, {
                "case": row["case"]+"_"+arm,
                "source_storage": arm,
                "graph": str(row["graph"]),
                "mapping": str(row["graph"][2]),
                "source_cpp_sha256": row[arm]["source_sha256"],
            }, summary=measured)
            outcomes.append({
                "case": row["case"], "arm": arm,
                "process_completed": receipt.get("completed_successfully"),
                "E2_qualified": qualified["native_e2_accepted"],
                "E3_qualified": qualified["native_e3_qualified"],
                "status": qualified["status"],
                "HLS_ESTIMATES_ONLY": True,
                "physical_E4": None,
            })
    return {
        **meta,
        "status": "ORIGINAL_VENDOR_ATTEMPTS_COMPLETED_QUALIFICATION_SEPARATE",
        "vendor_calls_made": len(outcomes),
        "outcomes": outcomes,
        "physical_E4": None,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--audit-native", action="store_true")
    p.add_argument("--execute-native", action="store_true")
    p.add_argument("--confirm-isolated-native", action="store_true")
    p.add_argument("--start-case", type=int, default=0)
    p.add_argument("--max-cases", type=int, default=1)
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--timeout-seconds", type=int, default=7200)
    args = p.parse_args(argv)
    report = (audit_native(study=args.study) if args.audit_native else
              execute(study=args.study, start_case=args.start_case,
                      max_cases=args.max_cases,
                      execute_native=args.execute_native,
                      confirm_isolated_native=args.confirm_isolated_native,
                      vitis_executable=args.vitis_executable,
                      timeout_seconds=args.timeout_seconds))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
