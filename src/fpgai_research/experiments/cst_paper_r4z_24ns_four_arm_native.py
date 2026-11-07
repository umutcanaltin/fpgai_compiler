"""R4z: new source-pinned 24ns four-arm Vitis E2/E3 study.

Plan/preparation/audit are vendor-free. Execution is explicitly double-opt-in
and one-time, against COPIED R4y input packages in a NEW create-only study.
The previous R4k/R4l/R4w native reports and attempt locks stay immutable.
An HLS II=1 pragma is a REQUEST, not measured DSP pipelining. E2 numerical
pass is necessary for E3; E3 is an HLS estimate, not routed E4 or board time.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil

from fpgai_research.experiments import (
    cst_paper_r4y_24ns_pipeline_source_prepare as r4y,
)
from fpgai_research.experiments.cst_nchw_hls_qualification import qualify_case
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

SCHEMA = "fpgai-research/R4z-24ns-four-arm-native-HLS-E2-E3/v1"
SOURCE_INPUTS = ("model.cpp", "tb.cpp", "run_hls.tcl", "manifest.json")
ATTEMPTS = ("vitis_invocation.json", "logs", "fpgai_hls_project")
RESOURCES = ("bram_18k", "dsp", "lut", "ff", "uram")
LOCK = "native_attempt_lock.json"


def _sha(p: Path) -> str:
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"R4z missing/symlinked source evidence: {p}")
    return sha256(p.read_bytes()).hexdigest()


def _json(p: Path) -> dict:
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 16_000_000:
        raise ValueError(f"R4z missing/unsafe receipt: {p}")
    parsed = json.loads(p.read_text())
    if not isinstance(parsed, dict):
        raise ValueError("R4z receipt is not an object")
    return parsed


def _source(*, source_study: str | Path, r4k_study: str | Path,
            original_r4d: str | Path, original_r4a: str | Path) -> dict:
    info = r4y.audit(
        study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    if (info.get("schema") != r4y.SCHEMA
            or info.get("status") != "SOURCE_ONLY_FOUR_ARM_PACKAGE_AUDITED_NOT_NATIVE"
            or info.get("cases") != list(r4y.CASES)
            or info.get("clock_period_ns") != r4y.CLOCK_NS
            or info.get("matched_testbench_and_tcl") is not True
            or info.get("new_vendor_calls_during_audit") != 0
            or info.get("originals_modified") is not False
            or not isinstance(info.get("source_revision"), str)
            or len(info["source_revision"]) != 40
            or set(info.get("source_sha256", {})) != set(r4y.CASES)):
        raise ValueError("R4z immutable R4y source-only audit unavailable")
    return info


def _expected(*, source_study: str | Path, r4k_study: str | Path,
              original_r4d: str | Path, original_r4a: str | Path,
              source: dict) -> dict:
    root = Path(source_study)
    return {
        "schema": SCHEMA,
        "status": "PREPARED_FRESH_FOUR_ARM_NATIVE_INPUTS_NOT_EXECUTED",
        "source_study": str(source_study),
        "r4k_study": str(r4k_study),
        "original_r4d": str(original_r4d),
        "original_r4a": str(original_r4a),
        "original_source_revision": source["source_revision"],
        "original_R4k_source_revision": source["original_R4k_source_revision"],
        "original_source_summary_sha256": _sha(root / "summary.json"),
        "original_source_inventory_sha256": _sha(root / "artifact_sha256.json"),
        "source_sha256": source["source_sha256"],
        "cases": list(r4y.CASES),
        "part": r4y.PART,
        "clock_period_ns": r4y.CLOCK_NS,
        "clock_uncertainty_ns": r4y.UNCERTAINTY_NS,
        "native_vendor_calls_at_preparation": 0,
        "original_evidence_modified": False,
        "E2": None,
        "E3": None,
        "E4": None,
        "board_qualified": False,
        "measured_latency_ns": None,
    }


def _source_kwargs(*, source_study: str | Path, r4k_study: str | Path,
                   original_r4d: str | Path, original_r4a: str | Path) -> dict:
    return {
        "source_study": source_study,
        "r4k_study": r4k_study,
        "original_r4d": original_r4d,
        "original_r4a": original_r4a,
    }


def _no_overlap(output: Path, roots: tuple[str | Path, ...]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("R4z native study must be create-only")
    target = output.resolve()
    for old in roots:
        path = Path(old)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("R4z original evidence directory missing/symlinked")
        frozen = path.resolve()
        if target == frozen or target in frozen.parents or frozen in target.parents:
            raise ValueError("R4z native root overlaps frozen evidence")


def prepare(*, output: str | Path, source_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    dest = Path(output)
    roots = (source_study, r4k_study, original_r4d, original_r4a)
    _no_overlap(dest, roots)
    audited = _source(**_source_kwargs(
        source_study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a))
    frozen = _expected(
        source_study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a,
        source=audited)
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for case in r4y.CASES:
            new = dest / case
            new.mkdir()
            old = Path(source_study) / case / "vitis_source"
            for name in SOURCE_INPUTS:
                before = old / name
                after = new / name
                checksum = _sha(before)
                shutil.copyfile(before, after)
                if _sha(after) != checksum:
                    raise ValueError("R4z copy did not preserve old source SHA")
            if _sha(new / "model.cpp") != audited["source_sha256"][case]:
                raise ValueError("R4z copied source differs from signed R4y arm")
            record = _json(new / "manifest.json")
            if (record.get("cpp_sha256") != audited["source_sha256"][case]
                    or record.get("clock_ns") != r4y.CLOCK_NS
                    or record.get("explicit_clock_uncertainty_ns") !=
                        r4y.UNCERTAINTY_NS
                    or record.get("part") != r4y.PART):
                raise ValueError("R4z target/source manifest mismatch")
        (dest / "summary.json").write_text(json.dumps(
            frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
        inventory = {"summary.json": _sha(dest / "summary.json")}
        for case in r4y.CASES:
            inventory.update({
                f"{case}/{name}": _sha(dest / case / name)
                for name in SOURCE_INPUTS
            })
        (dest / "input_sha256.json").write_text(json.dumps(
            inventory, indent=2, sort_keys=True) + "\n")
        return frozen
    except BaseException as exc:
        (dest / "FAILED.json").write_text(json.dumps({
            "schema": SCHEMA + "/source-preparation-failure",
            "error": type(exc).__name__ + ": " + str(exc)[:500],
            "new_native_calls": 0,
            "frozen_evidence_modified": False,
        }, indent=2, sort_keys=True) + "\n")
        raise


def _lock_expected(prepared: dict) -> dict:
    return {
        "schema": SCHEMA + "/one-time-attempt-lock/v1",
        "original_source_revision": prepared["original_source_revision"],
        "source_sha256": prepared["source_sha256"],
        "cases": list(r4y.CASES),
        "planned_new_vitis_invocations": len(r4y.CASES),
        "read_only_after_any_attempt": True,
    }


def audit_prepared(*, study: str | Path, source_study: str | Path,
                   r4k_study: str | Path, original_r4d: str | Path,
                   original_r4a: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4z prepared study root missing/symlinked")
    roots = (source_study, r4k_study, original_r4d, original_r4a)
    for old in roots:
        path = Path(old)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("R4z original evidence root missing/symlink")
        newer, frozen = root.resolve(), path.resolve()
        if newer == frozen or newer in frozen.parents or frozen in newer.parents:
            raise ValueError("R4z prepared study overlaps original evidence")
    audited = _source(**_source_kwargs(
        source_study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a))
    expected = _expected(
        source_study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a,
        source=audited)
    prepared = _json(root / "summary.json")
    if prepared != expected:
        raise ValueError("R4z native study's original source plan changed")
    inventory = _json(root / "input_sha256.json")
    keys = {"summary.json"} | {
        f"{case}/{name}" for case in r4y.CASES for name in SOURCE_INPUTS}
    if set(inventory) != keys or any(
            type(digest) is not str or len(digest) != 64 for digest in inventory.values()):
        raise ValueError("R4z native case input inventory invalid")
    for name, digest in inventory.items():
        if _sha(root / name) != digest:
            raise ValueError("R4z native source SHA inventory changed")
    for case in r4y.CASES:
        folder = root / case
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R4z case directory missing/symlinked")
        for name in SOURCE_INPUTS:
            if _sha(folder / name) != _sha(
                    Path(source_study) / case / "vitis_source" / name):
                raise ValueError("R4z staged native source no longer matches R4y")
    lockfile = root / LOCK
    attempted = lockfile.exists() or lockfile.is_symlink()
    if attempted:
        if _json(lockfile) != _lock_expected(prepared):
            raise ValueError("R4z native one-time attempt lock drift")
    else:
        # No vendor product or extra input is allowed before creating the lock.
        if {p.name for p in root.iterdir()} != (
                set(r4y.CASES) | {"summary.json", "input_sha256.json"}):
            raise ValueError("R4z unexplained pre-native study sidecar")
        for case in r4y.CASES:
            if {p.name for p in (root / case).iterdir()} != set(SOURCE_INPUTS):
                raise ValueError("R4z native artifacts exist without attempt lock")
    return {**prepared,
            "one_time_attempt_lock_present": attempted,
            "verified_input_sha256_count": len(inventory),
            "originals_modified": False,
            "vendor_calls_during_read_only_preflight": 0}


def _observation(folder: Path, *, case: str, source_sha: str) -> dict:
    present = folder / "vitis_invocation.json"
    if present.is_symlink():
        raise ValueError("R4z native receipt cannot be symlinked")
    if not present.is_file():
        return {
            "case": case, "native_attempt_receipt_present": False,
            "status": "NOT_RUN_OR_INCOMPLETE_NATIVE_ATTEMPT",
            "E2_numerical_qualified": False, "E3_HLS_qualified": False,
            "CSim_max_absolute_error": None,
            "CSim_failures": None,
            "HLS_worst_cycles": None, "HLS_top_level_II_max_cycles": None,
            "HLS_report_sha256": None, "HLS_estimated_clock_ns": None,
            "E3_requested_clock_cycle_product_ns_NOT_measured": None,
            "estimated_resources": {key: None for key in RESOURCES},
            "physical_E4": None, "real_measured_latency_ns": None,
            "warnings": [],
        }
    summary = summarize_case(str(folder))
    qualified = qualify_case(
        folder, {"case": "R4z_" + case, "graph": "R4y_fixed_fanout_MatMul",
                 "mapping": "R4y_storage_x_MAC_pipeline_II1",
                 "source_storage": case.split("__", 1)[0],
                 "source_cpp_sha256": source_sha}, summary=summary)
    if summary.get("source_sha256") != source_sha:
        raise ValueError("R4z current source and vendor observation differ")
    e2 = qualified.get("native_e2_accepted") is True
    e3 = e2 and qualified.get("native_e3_qualified") is True
    csim = summary.get("csim_receipt") or {}
    metrics = summary.get("metrics") or {}
    cycles = qualified.get("hls_worst_cycles") if e3 else None
    if e3 and (type(cycles) is not int or cycles <= 0):
        raise ValueError("R4z accepted E3 result lacks positive worst cycles")
    ii = metrics.get("ii_max") if e3 else None
    if type(ii) is str and ii.isdecimal():
        ii = int(ii)
    if type(ii) is not int or ii <= 0:
        ii = None
    return {
        "case": case, "native_attempt_receipt_present": True,
        "status": qualified.get("status"),
        "E2_numerical_qualified": e2,
        "E3_HLS_qualified": bool(e3),
        # Original fresh source-attested failure receipts are scientific
        # observations too; never hide an observed error on E2 rejection.
        "CSim_max_absolute_error": (
            csim.get("max_absolute_error") if csim else None),
        "CSim_failures": csim.get("failures") if csim else None,
        "HLS_worst_cycles": cycles,
        "HLS_top_level_II_max_cycles": ii,
        "HLS_report_sha256": (
            qualified.get("hls_report_sha256") if e3 else None),
        "HLS_estimated_clock_ns": (
            qualified.get("hls_estimated_clock_ns") if e3 else None),
        "E3_requested_clock_cycle_product_ns_NOT_measured": (
            cycles * r4y.CLOCK_NS if e3 else None),
        "estimated_resources": {
            key: (qualified.get("estimated_resources") or {}).get(key)
            if e3 else None for key in RESOURCES},
        "physical_E4": None,
        "real_measured_latency_ns": None,
        "warnings": qualified.get("warnings") or [],
    }


def audit_native(*, study: str | Path, source_study: str | Path,
                 r4k_study: str | Path, original_r4d: str | Path,
                 original_r4a: str | Path) -> dict:
    prepared = audit_prepared(
        study=study, source_study=source_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    root = Path(study)
    arms = [
        _observation(root / case, case=case,
                     source_sha=prepared["source_sha256"][case])
        for case in r4y.CASES
    ]
    completed = sum(arm["native_attempt_receipt_present"] for arm in arms)
    qualified_e2 = sum(arm["E2_numerical_qualified"] for arm in arms)
    qualified_e3 = sum(arm["E3_HLS_qualified"] for arm in arms)
    pairs = []
    for policy in r4y.PIPELINE:
        full = next(x for x in arms if x["case"] == "full_storage__" + policy)
        row = next(x for x in arms if x["case"] == "row_storage__" + policy)
        both = full["E3_HLS_qualified"] and row["E3_HLS_qualified"]
        pairs.append({
            "pipeline_policy": policy,
            "equal_target_clock_ns": r4y.CLOCK_NS,
            "both_E2_qualified": (full["E2_numerical_qualified"] and
                                  row["E2_numerical_qualified"]),
            "both_E3_HLS_qualified": bool(both),
            "HLS_worst_cycles_delta_row_minus_full": (
                row["HLS_worst_cycles"] - full["HLS_worst_cycles"]
                if both else None),
            "HLS_cycle_product_ns_delta_NOT_measured": (
                row["E3_requested_clock_cycle_product_ns_NOT_measured"] -
                full["E3_requested_clock_cycle_product_ns_NOT_measured"]
                if both else None),
            "routed_E4_qualified": False,
            "measured_FPGA_latency_delta_ns": None,
        })
    # Within-storage intervention effects preserve the factorial's other
    # axis. Only E3-qualified source-attested arms enter quantitative HLS
    # comparisons. Invalid/missing original observations stay NULL.
    pipeline_effects = []
    for storage in r4y.STORAGE:
        base = next(x for x in arms if x["case"] == storage + "__baseline")
        requested = next(x for x in arms
                         if x["case"] == storage + "__producer_mac_ii1_requested")
        pair_valid = base["E3_HLS_qualified"] and requested["E3_HLS_qualified"]
        pipeline_effects.append({
            "storage": storage, "both_E3_HLS_qualified": bool(pair_valid),
            "HLS_worst_cycles_delta_requested_minus_baseline": (
                requested["HLS_worst_cycles"] - base["HLS_worst_cycles"]
                if pair_valid else None),
            "HLS_clock_product_ns_delta_NOT_measured": (
                requested["E3_requested_clock_cycle_product_ns_NOT_measured"] -
                base["E3_requested_clock_cycle_product_ns_NOT_measured"]
                if pair_valid else None),
            "HLS_resource_delta_requested_minus_baseline": {
                key: (requested["estimated_resources"][key] -
                      base["estimated_resources"][key]
                      if pair_valid and
                      type(requested["estimated_resources"][key]) in (int, float)
                      and type(base["estimated_resources"][key]) in (int, float)
                      else None)
                for key in RESOURCES
            },
            "verified_producer_loop_II_cycles": None,
            "measured_FPGA_latency_delta_ns": None,
        })
    interaction_valid = all(x["both_E3_HLS_qualified"] for x in pipeline_effects)
    full_effect, row_effect = pipeline_effects
    interaction = {
        "all_four_E3_qualified": bool(interaction_valid),
        "pipeline_x_storage_interaction_HLS_cycles": (
            row_effect["HLS_worst_cycles_delta_requested_minus_baseline"] -
            full_effect["HLS_worst_cycles_delta_requested_minus_baseline"]
            if interaction_valid else None),
        "pipeline_x_storage_interaction_HLS_clock_product_ns_NOT_measured": (
            row_effect["HLS_clock_product_ns_delta_NOT_measured"] -
            full_effect["HLS_clock_product_ns_delta_NOT_measured"]
            if interaction_valid else None),
        "physical_interaction_latency_ns": None,
    }
    return {
        "schema": SCHEMA,
        "status": ("FOUR_NATIVE_RECEIPTS_READ_ONLY_HLS_REVIEW"
                   if completed == 4 else "PARTIAL_OR_UNATTEMPTED_NATIVE_REVIEW"),
        "original_source_revision": prepared["original_source_revision"],
        "one_time_attempt_lock_present": prepared["one_time_attempt_lock_present"],
        "cases_expected": len(r4y.CASES),
        "native_receipts_present": completed,
        "E2_qualified_cases": qualified_e2,
        "E3_qualified_cases": qualified_e3,
        "observations": arms,
        "same_clock_storage_pairs": pairs,
        "within_storage_pipeline_effects": pipeline_effects,
        "four_arm_factorial_interaction": interaction,
        "new_vendor_calls_during_audit": 0,
        "original_evidence_modified": False,
        "physical_E4": None,
        "board_qualified": False,
        "board_measured_latency_ns": None,
        "limitations": [
            "E2 is fresh source-attributed numerical CSim; E3 is HLS estimates only",
            "24ns is an imposed requested clock, not independently measured clock",
            "HLS top-level II is not proof of producer MAC-loop II or DSP pipeline registers",
            "HLS worst cycles times requested clock is NOT measured core/board latency",
            "Routed Vivado setup/hold, DRC, board/DMA/throughput remain UNKNOWN",
        ],
    }


def execute(*, study: str | Path, source_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, execute_native: bool = False,
            confirm_isolated_native: bool = False,
            vitis_executable: str = "vitis_hls",
            timeout_seconds: int = 7200) -> dict:
    kw = dict(study=study, source_study=source_study, r4k_study=r4k_study,
              original_r4d=original_r4d, original_r4a=original_r4a)
    prepared = audit_prepared(**kw)
    if not execute_native:
        return {
            "schema": SCHEMA, "status": "DRY_RUN_NO_VENDOR_CALLS",
            "cases": list(r4y.CASES), "planned_vendor_calls": 4,
            "native_vendor_calls": 0,
            "one_time_attempt_lock_present":
                prepared["one_time_attempt_lock_present"],
            "original_evidence_modified": False,
        }
    if not confirm_isolated_native:
        raise ValueError("R4z requires --confirm-isolated-native to run Vitis")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
        raise ValueError("R4z native timeout must be integer within [1,86400]")
    if prepared["one_time_attempt_lock_present"]:
        raise FileExistsError("R4z native study already attempted; no rerun")
    executable = shutil.which(vitis_executable)
    if executable is None:
        raise FileNotFoundError("R4z Vitis HLS executable unavailable")
    root = Path(study)
    for case in r4y.CASES:
        folder = root / case
        for artifact in ATTEMPTS:
            file = folder / artifact
            if file.exists() or file.is_symlink():
                raise FileExistsError(
                    "R4z original native case already attempted: " + str(file))
    # This lock precedes ANY vendor call. An exception/timeout still forbids
    # reusing this study root, preventing partial results from being cherry-picked.
    with (root / LOCK).open("x") as stream:
        stream.write(json.dumps(_lock_expected(prepared),
                                indent=2, sort_keys=True) + "\n")
    outputs = []
    for case in r4y.CASES:
        invocation = run_vitis(
            root / case, executable=executable, timeout_seconds=timeout_seconds)
        outputs.append({
            "case": case,
            "completed_successfully":
                invocation.get("completed_successfully") is True,
        })
    return {
        "schema": SCHEMA,
        "status": "FOUR_VENDOR_ATTEMPTS_RECORDED_RESULTS_REQUIRE_READ_ONLY_AUDIT",
        "one_time_attempt_lock_present": True,
        "native_vendor_calls": len(outputs),
        "attempt_outcomes": outputs,
        "E2_E3_HLS_results_are_qualified": False,
        "real_measured_latency_ns": None,
        "physical_E4": None,
        "original_evidence_modified": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-output")
    mode.add_argument("--audit-native", action="store_true")
    parser.add_argument("--study")
    parser.add_argument("--source-study",
                        default="build/r4y_24ns_four_arm_sources_20261106_01")
    parser.add_argument("--r4k-study",
                        default="build/r4k_source_schedule_matched_storage_20261106_01")
    parser.add_argument("--original-r4d",
                        default="build/r4d_numeric_format_source_20261106_01")
    parser.add_argument("--original-r4a",
                        default="build/r4a_row_retained_paper_20261106_01")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--confirm-isolated-native", action="store_true")
    parser.add_argument("--vitis-executable", default="vitis_hls")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args(argv)
    source_args = dict(
        source_study=args.source_study, r4k_study=args.r4k_study,
        original_r4d=args.original_r4d, original_r4a=args.original_r4a)
    if args.prepare_output:
        if args.execute_native or args.confirm_isolated_native:
            parser.error("R4z source preparation never executes vendor tools")
        out = prepare(output=args.prepare_output, **source_args)
    else:
        if not args.study:
            parser.error("R4z --study required for dry-run/audit/native run")
        if args.audit_native:
            if args.execute_native or args.confirm_isolated_native:
                parser.error("R4z read-only audit cannot execute native tools")
            out = audit_native(study=args.study, **source_args)
        else:
            out = execute(
                study=args.study, **source_args,
                execute_native=args.execute_native,
                confirm_isolated_native=args.confirm_isolated_native,
                vitis_executable=args.vitis_executable,
                timeout_seconds=args.timeout_seconds)
    print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
