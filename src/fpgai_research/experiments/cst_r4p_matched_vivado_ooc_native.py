"""R4p: create-only native Vivado OOC execution copies + strict routed audit.

Never run Vivado against R4o, R4k, or the original Vitis folders. R4p
re-stages signed R4o inputs into NEW cases compatible with the existing
Vivado runner's original-RTL/ROM provenance layout. This uses deliberate
byte-identical ROM copies at the *runner-checked* cwd and at the *Tcl-
consumed* rtl/ working directory. Both are attested before/after.

Preparation and auditing are vendor-free. Native execution requires two
flags. Original R4l native Vitis results remain immutable. Routed OOC
timing and DRC are not verified KV260 board deployment or achieved Fmax.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import shutil

from fpgai_research.experiments import cst_r4o_matched_vivado_ooc_prepare as r4o
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.vivado.runner import run_vivado
from fpgai_research.toolchains.vivado.rtl_inputs import compare_staged_roms

SCHEMA = "fpgai-research/R4p-isolated-source-pinned-routed-OOC/v1"
ARMS = r4o.ARMS
_INPUTS = ("run_vivado.tcl",)
_ATTEMPT = ("logs", "reports", "outputs", "vivado_invocation.json",
            ".Xil", "vivado.jou", "vivado.log")
_REPORTS = ("route_status.rpt", "timing_summary.rpt",
            "utilization.rpt", "drc.rpt")
_SHA_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
_RESOURCE = ("lut", "ff", "dsp", "bram", "uram")


def _sha(file: Path) -> str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("R4p source file missing/symlinked: " + str(file))
    return sha256(file.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("R4p report or original inventory missing/unsafe: " + str(path))
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("R4p JSON evidence must be an object")
    return result


def _root(path: str | Path) -> Path:
    p = Path(path)
    if p.is_symlink() or not p.is_dir():
        raise ValueError("R4p original/prepared study directory missing/unsafe")
    return p


def _original_r4o(*, r4o_study: str | Path, r4k_study: str | Path,
                   original_r4d: str | Path, original_r4a: str | Path) -> dict:
    """Recheck original R4l E2/E3 -> R4n RTL -> R4o copy and SHA chain."""
    source = r4o.inspect(r4k_study=r4k_study,
                         original_r4d=original_r4d, original_r4a=original_r4a)
    root = _root(r4o_study)
    original = _json(root / "summary.json")
    manifest = _json(root / "artifact_sha256.json")
    if original != {
            **source, "status": "SOURCE_ONLY_VIVADO_OOC_PREPARED_NOT_EXECUTED",
            "new_vivado_calls": 0, "physical_E4": None}:
        raise ValueError("R4p original R4o source-only qualification changed")
    expected = {"summary.json"}
    for arm in ARMS:
        inputs = source["arms"][arm]
        expected.add(f"{arm}/run_vivado.tcl")
        expected.update(f"{arm}/rtl/{name}"
                        for name in (*inputs["rtl_sha256"], *inputs["rom_sha256"]))
    if (set(manifest) != expected
            or any(type(k) is not str or type(v) is not str
                   or not _SHA_PATTERN.fullmatch(v)
                   for k,v in manifest.items())):
        raise ValueError("R4p R4o immutable source-only file inventory incomplete")
    for name, digest in manifest.items():
        if _sha(root / name) != digest:
            raise ValueError("R4p R4o source-only SHA inventory mismatch")
    for arm in ARMS:
        group = source["arms"][arm]
        script = root / arm / "run_vivado.tcl"
        if script.read_text() != r4o._tcl(
                rtl_names=sorted(group["rtl_sha256"])):
            raise ValueError("R4p original R4o target clock/part/Tcl source drift")
        rtldir = root / arm / "rtl"
        if rtldir.is_symlink() or not rtldir.is_dir():
            raise ValueError("R4p original R4o RTL/ROM input folder missing")
        if set(p.name for p in rtldir.iterdir()) != (
                set(group["rtl_sha256"]) | set(group["rom_sha256"])):
            raise ValueError("R4p original R4o staged source file population changed")
    return source


def prepare(*, r4o_study: str | Path, r4k_study: str | Path,
            original_r4d: str | Path, original_r4a: str | Path,
            output: str | Path) -> dict:
    """Create new source-only R4p execution cases; zero Vivado calls."""
    original = _original_r4o(
        r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    dest = Path(output)
    resolved = dest.resolve()
    roots = tuple(Path(p).resolve() for p in (
        r4o_study, r4k_study, original_r4d, original_r4a))
    if any(resolved == p or p in resolved.parents
           or resolved in p.parents for p in roots):
        raise ValueError("R4p output must be outside and distinct from original evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4p native execution study output must be create-only")
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for arm in ARMS:
            stage = Path(r4o_study) / arm
            group = original["arms"][arm]
            case = dest / arm
            case.mkdir()
            canonical = (case / "fpgai_hls_project" / "solution1"
                         / "syn" / "verilog")
            canonical.mkdir(parents=True)
            native = case / "vivado"
            native.mkdir()
            consumed = native / "rtl"
            consumed.mkdir()
            # Canonical run_tool source provenance: original RTL/ROM copies
            # live at case/fpgai_hls_project/solution1/syn/verilog.
            # Vivado Tcl consumes *another* exact copy in vivado/rtl.
            # run_tool checks local ROMs at vivado/*.dat too.
            allfiles = {**group["rtl_sha256"], **group["rom_sha256"]}
            for name, digest in sorted(allfiles.items()):
                source = stage / "rtl" / name
                for target in (canonical / name, consumed / name):
                    shutil.copyfile(source, target, follow_symlinks=False)
                    if _sha(target) != digest:
                        raise ValueError("R4p copied original RTL or ROM SHA mismatch")
                if name in group["rom_sha256"]:
                    target = native / name
                    shutil.copyfile(source, target, follow_symlinks=False)
                    if _sha(target) != digest:
                        raise ValueError("R4p runner ROM SHA copy mismatch")
            script = stage / "run_vivado.tcl"
            shutil.copyfile(script, native / "run_vivado.tcl",
                            follow_symlinks=False)
            if _sha(script) != _sha(native / "run_vivado.tcl"):
                raise ValueError("R4p Vivado source-only Tcl copy mismatch")
        report = {
            "schema": SCHEMA,
            "status": "R4P_NATIVE_CASES_PREPARED_NOT_EXECUTED",
            "original_R4o_schema": original["schema"],
            "original_R4k_source_revision":
                original["original_R4k_source_revision"],
            "target_part": original["target_part"],
            "requested_clock_ns": original["requested_clock_ns"],
            "source_clock_uncertainty_ns": original["source_clock_uncertainty_ns"],
            "top": original["top"],
            "original_R4o_study": str(Path(r4o_study)),
            "native_vendor_calls": 0,
            "paired_native_arms": len(ARMS),
            "physical_E4": None,
            "board_qualified": False,
        }
        (dest / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        hashes = {
            str(p.relative_to(dest)): _sha(p)
            for p in sorted(dest.rglob("*")) if p.is_file()
        }
        (dest / "artifact_sha256.json").write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        return report
    except BaseException as exc:
        (dest / "FAILED.json").write_text(json.dumps({
            "exception_class": type(exc).__name__,
            "reason": str(exc)[:2048],
            "source_evidence_modified": False,
            "native_vendor_calls": 0,
        }, indent=2, sort_keys=True) + "\n")
        raise


def audit_prepared(*, study: str | Path, r4o_study: str | Path,
                   r4k_study: str | Path, original_r4d: str | Path,
                   original_r4a: str | Path) -> dict:
    """Verify every original + copied staged byte before native execution."""
    r4p = _root(study)
    original = _original_r4o(
        r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    if any(r4p.resolve() == Path(x).resolve()
           or r4p.resolve() in Path(x).resolve().parents
           or Path(x).resolve() in r4p.resolve().parents
           for x in (r4o_study, r4k_study, original_r4d, original_r4a)):
        raise ValueError("R4p execution study overlaps immutable source evidence")
    receipt = _json(r4p / "summary.json")
    if (receipt.get("schema") != SCHEMA
            or receipt.get("status") != "R4P_NATIVE_CASES_PREPARED_NOT_EXECUTED"
            or receipt.get("original_R4o_schema") != original["schema"]
            or receipt.get("original_R4k_source_revision") !=
                original["original_R4k_source_revision"]
            or receipt.get("target_part") != original["target_part"]
            or receipt.get("top") != original["top"]
            or receipt.get("requested_clock_ns") != original["requested_clock_ns"]
            or receipt.get("source_clock_uncertainty_ns") !=
                original["source_clock_uncertainty_ns"]
            or receipt.get("paired_native_arms") != len(ARMS)
            or receipt.get("native_vendor_calls") != 0
            or receipt.get("physical_E4") is not None
            or receipt.get("board_qualified") is not False):
        raise ValueError("R4p original physical source-only receipt mismatch")
    hashes = _json(r4p / "artifact_sha256.json")
    expected = {"summary.json"}
    for arm in ARMS:
        group = original["arms"][arm]
        for name in (*group["rtl_sha256"], *group["rom_sha256"]):
            expected.update((
                f"{arm}/fpgai_hls_project/solution1/syn/verilog/{name}",
                f"{arm}/vivado/rtl/{name}",
            ))
            if name in group["rom_sha256"]:
                expected.add(f"{arm}/vivado/{name}")
        expected.add(f"{arm}/vivado/run_vivado.tcl")
    if set(hashes) != expected:
        raise ValueError("R4p new original source-only SHA inventory incomplete")
    for name, digest in hashes.items():
        if (type(name) is not str or type(digest) is not str
                or not _SHA_PATTERN.fullmatch(digest)
                or _sha(r4p / name) != digest):
            raise ValueError("R4p staged original SHA or source-manifest mismatch")
    for arm in ARMS:
        src = original["arms"][arm]
        expected_all = {**src["rtl_sha256"], **src["rom_sha256"]}
        for name, digest in expected_all.items():
            for prefix in (
                    f"{arm}/fpgai_hls_project/solution1/syn/verilog/",
                    f"{arm}/vivado/rtl/"):
                if _sha(r4p / (prefix + name)) != digest:
                    raise ValueError("R4p original Verilog/ROM lineage diverged")
            if name in src["rom_sha256"] and (
                    _sha(r4p / arm / "vivado" / name) != digest):
                raise ValueError("R4p runner local ROM asset changed")
        if _sha(r4p / arm / "vivado" / "run_vivado.tcl") != _sha(
                Path(r4o_study) / arm / "run_vivado.tcl"):
            raise ValueError("R4p original clock/part/ROM Tcl digest changed")
        if not compare_staged_roms(
                r4p / arm / "vivado", src["rom_sha256"])["matches_original_roms"]:
            raise ValueError("R4p runner ROM assets not identical to source")
    return {
        "schema": SCHEMA,
        "source_gate": "ALL_R4L_R4O_R4P_SOURCE_BYTES_SHA_ATTESTED",
        "source_revision": original["original_R4k_source_revision"],
        "both_E2_E3_qualified_in_frozen_R4l": True,
        "source_Vivado_inputs_sha_attested": True,
        "new_vendor_calls_during_audit": 0,
        "physical_E4": None,
        "board_qualified": False,
    }


def _attempted(case: Path) -> bool:
    native = case / "vivado"
    return any((native / name).exists() or (native / name).is_symlink()
               for name in _ATTEMPT)


def execute(*, study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, execute_native: bool = False,
            confirm_ooc_native: bool = False, executable: str = "vivado",
            timeout_seconds: int = 14400) -> dict:
    meta = audit_prepared(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    if not execute_native:
        return {**meta, "status": "DRY_RUN_NO_VENDOR_CALLS",
                "planned_vivado_calls": 2, "actual_vivado_calls": 0}
    if not confirm_ooc_native:
        raise ValueError("R4p real Vivado OOC execution requires explicit confirmation")
    if type(timeout_seconds) is not int or not (1 <= timeout_seconds <= 172800):
        raise ValueError("R4p bounded Vivado timeout required")
    binary = shutil.which(executable)
    if binary is None:
        raise FileNotFoundError("R4p Vivado executable missing")
    root = Path(study)
    if any(_attempted(root / arm) for arm in ARMS):
        raise FileExistsError("R4p original OOC source arm already attempted; preserve native evidence")
    observations = []
    for arm in ARMS:
        case = root / arm
        native = case / "vivado"
        # Original source receipt and both independent ROM copies have
        # already been checked before launching either arm.
        outcome = run_vivado(
            native, executable=binary, timeout_seconds=timeout_seconds,
            allow_verified_readmem_post_absence=False)
        # Even a 0 exit code does not establish route/timing/E4.
        success = outcome.get("completed_successfully") is True
        observations.append({
            "arm": arm,
            "tool_process_completed_successfully": success,
            "native_receipt_path": str(native / "vivado_invocation.json"),
        })
        if not success:
            # Preserve the partial/failed case and leave the second arm
            # untouched. Never resume automatically after a vendor error.
            break
    return {
        **meta,
        "status": (
            "R4P_NATIVE_ATTEMPT_FAILED_PRESERVE_ALL_OUTPUTS"
            if len(observations) != 2 else
            "R4P_NATIVE_ATTEMPTS_RECORDED_AUDIT_REQUIRED"),
        "native_vendor_calls_executed": len(observations),
        "observations": observations,
        "physical_E4": None,
        "board_qualified": False,
    }


def _arm_observation(case: Path, *, source: dict) -> dict:
    native = case / "vivado"
    receipt_path = native / "vivado_invocation.json"
    record = {
        "arm": case.name,
        "status": "UNKNOWN_NO_OR_INCOMPLETE_NATIVE_PHYSICAL_EVIDENCE",
        "vendor_invocation_present": receipt_path.is_file(),
        "vendor_completed": False,
        "original_RTL_ROM_stable": None,
        "native_routed_observation_valid": False,
        "strict_clean_DRC_E4": False,
        "drc_warning_only_routed_observation": False,
        "routed_wns_ns": None,
        "routed_utilization": {k: None for k in _RESOURCE},
        "Drc": None,
        "source_R4l_HLS_estimated_BRAM18K": source["HLS_estimated_BRAM18K"],
        "source_cpp_sha256": source["source_cpp_sha256"],
        "original_RTL_file_count": len(source["rtl_sha256"]),
        "original_ROM_file_count": len(source["rom_sha256"]),
        "fresh_original_report_sha256": None,
        "board_qualified": False,
    }
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return record
    receipt = _json(receipt_path)
    canonical = {**source["rtl_sha256"], **source["rom_sha256"]}
    # Validate the exact vendor invocation, working directory and logs,
    # not only a process-success Boolean in a JSON file. The generic
    # runner writes these witnesses after each real vendor attempt.
    script_sha = _sha(native / "run_vivado.tcl")
    argv = receipt.get("argv")
    exe = receipt.get("executable")
    expected_flags = ["-mode", "batch", "-source",
                      "run_vivado.tcl", "-notrace"]
    invoke_ok = (
        type(exe) is str and bool(exe)
        and type(argv) is list and len(argv) == 6
        and type(argv[0]) is str and bool(argv[0])
        and argv[1:] == expected_flags
        and Path(argv[0]).resolve() == Path(exe).resolve()
        and receipt.get("cwd") == str(native.resolve())
        and receipt.get("input_hashes_before_execution") ==
            {"tcl": script_sha}
        and receipt.get("input_hashes_after_execution") ==
            {"tcl": script_sha}
        and type(receipt.get("started_utc")) is str
        and bool(receipt["started_utc"])
        and type(receipt.get("duration_seconds")) in (int, float)
        and isfinite(receipt["duration_seconds"])
        and receipt["duration_seconds"] >= 0
    )
    logs_ok = all(
        type(receipt.get(key)) is str
        and _SHA_PATTERN.fullmatch(receipt[key]) is not None
        and (native / "logs" / name).is_file()
        and _sha(native / "logs" / name) == receipt[key]
        for key, name in (
            ("stdout_sha256", "vivado.stdout.log"),
            ("stderr_sha256", "vivado.stderr.log"),
        )
    )
    # run_tool receipts are authoritative for original source and local
    # ROM integrity only if the exact expected hashes are present.
    valid = (
        invoke_ok and logs_ok
        and
        receipt.get("schema") == "fpgai-research/vendor-invocation/v1"
        and receipt.get("tool") == "vivado"
        and receipt.get("completed_successfully") is True
        and receipt.get("returncode") == 0
        and receipt.get("timed_out") is False
        and receipt.get("inputs_stable_during_execution") is True
        and receipt.get("rom_inputs_stable_during_execution") is True
        and receipt.get("staged_rom_inputs_stable_during_execution") is True
        and receipt.get("rtl_input_sha256") == source["rtl_sha256"]
        and receipt.get("rom_input_sha256") == source["rom_sha256"]
        and receipt.get("script_sha256") == script_sha
        and receipt.get("staged_rom_input_integrity_before", {}).get(
            "matches_original_roms") is True
        and receipt.get("staged_rom_input_integrity_after", {}).get(
            "matches_original_roms") is True
        and receipt.get("allow_verified_readmem_post_absence") is False
    )
    record["vendor_completed"] = receipt.get("completed_successfully") is True
    record["original_RTL_ROM_stable"] = bool(valid)
    if not valid:
        record["status"] = "REJECTED_VENDOR_OR_SOURCE_PROVENANCE"
        return record
    # Runner compares only its checked cwd ROM files. Tcl consumes copies
    # under ./rtl; verify BOTH independent input locations and reject drift.
    for name, digest in canonical.items():
        if (_sha(native / "rtl" / name) != digest
                or _sha(case / "fpgai_hls_project" / "solution1"
                        / "syn" / "verilog" / name) != digest):
            record["status"] = "REJECTED_ORIGINAL_TCL_CONSUMED_SOURCE_DRIFT"
            return record
    if any(_sha(native / name) != digest
           for name, digest in source["rom_sha256"].items()):
        record["status"] = "REJECTED_RUNNER_ROM_DRIFT"
        return record
    reports = native / "reports"
    fresh = receipt.get("fresh_report_sha256")
    if not isinstance(fresh, dict) or any(
            not isinstance(fresh.get(name), str)
            or not _SHA_PATTERN.fullmatch(fresh[name])
            or not (reports / name).is_file()
            or _sha(reports / name) != fresh[name]
            for name in _REPORTS):
        record["status"] = "UNKNOWN_MISSING_UNATTESTED_OR_STALE_VIVADO_REPORTS"
        return record
    record["fresh_original_report_sha256"] = {
        name: fresh[name] for name in _REPORTS}
    try:
        physical = parse_vivado_reports(native)
    except (ValueError, FileNotFoundError) as exc:
        record["status"] = "UNKNOWN_INCOMPLETE_ROUTE_TIMING_OR_DRC"
        record["parse_issue"] = type(exc).__name__ + ": " + str(exc)[:400]
        return record
    resources = physical["utilization"]
    if not (physical.get("routed_nets_complete") is True
            and physical["timing"]["timing_met"] is True
            and type(physical["timing"]["wns_ns"]) in (int, float)
            and isfinite(physical["timing"]["wns_ns"])
            and all(type(resources.get(key)) in (int, float)
                    and isfinite(resources[key]) and resources[key] >= 0
                    for key in ("lut", "ff", "dsp", "bram"))):
        record["status"] = "UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCE_OR_TIMING"
        return record
    drc = physical["drc"]
    if drc["summary_verified"] is not True or type(drc["violations"]) is not int:
        record["status"] = "UNKNOWN_DRC_NOT_VERIFIABLY_SUMMARIZED"
        return record
    strict = drc["clean"] is True
    record.update({
        "status": (
            "SOURCE_ATTRIBUTED_CLEAN_DRC_ROUTED_OOC_ONLY" if strict
            else "SOURCE_ATTRIBUTED_ROUTED_NONCLEAN_DRC_OBSERVATION_ONLY"),
        "native_routed_observation_valid": True,
        "strict_clean_DRC_E4": strict,
        "drc_warning_only_routed_observation": (
            not strict and drc.get("critical_warnings") == 0),
        "drc_errors_or_critical_warnings_present": (
            drc.get("critical_warnings", 0) > 0),
        "routed_wns_ns": physical["timing"]["wns_ns"],
        "routed_utilization": {
            k: resources.get(k) for k in _RESOURCE
        },
        "Drc": drc,
        "board_qualified": False,
    })
    return record


def audit_native(*, study: str | Path, r4o_study: str | Path,
                 r4k_study: str | Path, original_r4d: str | Path,
                 original_r4a: str | Path) -> dict:
    meta = audit_prepared(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    original = _original_r4o(
        r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    root = Path(study)
    arms = [
        _arm_observation(root / name, source=original["arms"][name])
        for name in ARMS
    ]
    first, second = arms
    descriptive = first["native_routed_observation_valid"] and (
        second["native_routed_observation_valid"])
    strict = first["strict_clean_DRC_E4"] and second["strict_clean_DRC_E4"]
    delta = {
        key: (second["routed_utilization"][key] -
              first["routed_utilization"][key]
              if descriptive and first["routed_utilization"][key] is not None
              and second["routed_utilization"][key] is not None else None)
        for key in _RESOURCE
    }
    wn = (second["routed_wns_ns"] - first["routed_wns_ns"]
          if descriptive else None)
    return {
        **meta,
        "status": ("STRICT_CLEAN_DRC_BOTH_OOC_ROUTED" if strict
                   else "ROUTED_NONCLEAN_DRC_PAIR" if descriptive
                   else "UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR"),
        "arms": arms,
        "original_native_invocation_receipts_present": sum(
            arm["vendor_invocation_present"] for arm in arms),
        "both_source_attributed_routed_observations_valid": descriptive,
        "both_strict_clean_DRC_OOC_E4": strict,
        "descriptive_physical_resource_delta_row_minus_full": delta,
        "strict_clean_DRC_resource_delta_row_minus_full":
            delta if strict else {key: None for key in _RESOURCE},
        "descriptive_WNS_delta_row_minus_full_ns": wn,
        "strict_clean_DRC_WNS_delta_ns": wn if strict else None,
        "HLS_estimated_BRAM18K_tie": (
            first["source_R4l_HLS_estimated_BRAM18K"]
            == second["source_R4l_HLS_estimated_BRAM18K"]),
        "new_vivado_calls_during_audit": 0,
        "physical_achieved_board_Fmax": None,
        "board_qualified": False,
        "original_sources_modified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", help="Prepared R4p study (not needed with --prepare-output)")
    parser.add_argument("--r4o-study", required=True)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--prepare-output")
    parser.add_argument("--audit-native", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--confirm-ooc-native", action="store_true")
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args(argv)
    if not args.prepare_output and not args.study:
        parser.error("--study is required for dry-run, execution or native audit")
    opts = dict(
        r4o_study=args.r4o_study, r4k_study=args.r4k_study,
        original_r4d=args.original_r4d, original_r4a=args.original_r4a)
    if args.prepare_output:
        if args.execute_native or args.audit_native:
            parser.error("R4p preparation is source-only; cannot execute/audit simultaneously")
        result = prepare(**opts, output=args.prepare_output)
    elif args.audit_native:
        if args.execute_native:
            parser.error("R4p audited native output cannot also be executed")
        result = audit_native(study=args.study, **opts)
    else:
        result = execute(
            study=args.study, **opts,
            execute_native=args.execute_native,
            confirm_ooc_native=args.confirm_ooc_native,
            executable=args.vivado, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
