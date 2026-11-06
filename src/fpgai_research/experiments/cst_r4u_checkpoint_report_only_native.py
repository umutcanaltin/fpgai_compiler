"""R4u: isolated one-time Vivado REPORT-ONLY on copied routed checkpoints.

R4u NEVER re-runs R4p or synthesis/placement/route. R4t's two post-hoc
SHA-pinned DCP copies are again copied into a NEW independent R4u study.
The original R4p vendor receipt did NOT contemporaneously sign DCP bytes.
Report-only native invocation is explicit, one-time, atomically locked,
and fail-closed. Native result is supplementary checkpoint forensics,
NEVER physical clean-DRC E4 or a board frequency measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from fpgai_research.experiments import cst_r4t_routed_checkpoint_report_only_prepare as r4t
from fpgai_research.experiments import cst_r4s_original_timing_drc_context as r4s
from fpgai_research.toolchains.vivado.report import (
    _worst_negative_slack, _drc_summary,
)

SCHEMA = "fpgai-research/R4u-copied-DCP-one-time-report-only-native/v1"
ARMS = r4t.ARMS
REPORT_NAMES = ("max_delay_paths.rpt", "timing_summary.rpt", "drc.rpt")
SCRIPT = "report_only.tcl"
CHECKPOINT = "input/routed.dcp"
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 14400
SHA_CHARS = frozenset("0123456789abcdef")


def _digest(path: Path, *, size_limit: int | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4u input/report absent or symlinked: " + str(path))
    if size_limit is not None and not (0 < path.stat().st_size <= size_limit):
        raise ValueError("R4u input/report size invalid: " + str(path))
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError("R4u original signed JSON receipt missing or unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("R4u JSON receipt not an object")
    return value


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= SHA_CHARS


def _original_r4t(*, r4t_study: str | Path, study: str | Path,
                  r4o_study: str | Path, r4k_study: str | Path,
                  original_r4d: str | Path, original_r4a: str | Path) -> dict:
    original = r4t.inspect(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    root = Path(r4t_study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4u original R4t source-only root unsafe")
    expected_summary = {
        **original,
        "status": "R4T_ISOLATED_CHECKPOINT_REPORT_ONLY_SOURCE_PREPARED",
        "planned_new_report_only_vivado_calls": 2,
        "new_native_vivado_calls": 0,
        "strict_physical_E4": None,
    }
    if _json(root / "summary.json") != expected_summary:
        raise ValueError("R4u original R4t source-only summary changed")
    hashes = _json(root / "artifact_sha256.json")
    expected_files = {"summary.json"} | {
        f"{arm}/{name}"
        for arm in ARMS for name in (CHECKPOINT, SCRIPT)
    }
    if set(hashes) != expected_files or not all(
        _is_sha(digest) for digest in hashes.values()
    ):
        raise ValueError("R4u original R4t source-only file population changed")
    for name, digest in hashes.items():
        if _digest(root / name) != digest:
            raise ValueError("R4u original R4t source-only SHA changed")
    for arm in ARMS:
        item = original["arms"][arm]
        if (_digest(root / arm / CHECKPOINT) !=
                item["posthoc_original_checkpoint_sha256"]
                or _digest(root / arm / SCRIPT) != sha256(
                    r4t._script().encode()).hexdigest()
                or item["original_dcp_sha_contemporaneous_in_R4p_vendor_receipt"]
                    is not False):
            raise ValueError("R4u original posthoc DCP or inert Tcl mismatched")
        if (root / arm / "input").is_symlink():
            raise ValueError("R4u original R4t input folder symlinked")
        if {p.name for p in (root / arm / "input").iterdir()} != {"routed.dcp"}:
            raise ValueError("R4u original R4t checkpoint input population changed")
    return original


def _roots(*, r4t_study, study, r4o_study, r4k_study,
           original_r4d, original_r4a):
    return tuple(Path(p).resolve() for p in (
        r4t_study, study, r4o_study, r4k_study,
        original_r4d, original_r4a))


def prepare(*, r4t_study: str | Path, study: str | Path,
            r4o_study: str | Path, r4k_study: str | Path,
            original_r4d: str | Path, original_r4a: str | Path,
            output: str | Path) -> dict:
    args = dict(r4t_study=r4t_study, study=study,
                r4o_study=r4o_study, r4k_study=r4k_study,
                original_r4d=original_r4d, original_r4a=original_r4a)
    original = _original_r4t(**args)
    out = Path(output)
    dest = out.resolve()
    if any(dest == p or p in dest.parents or dest in p.parents
           for p in _roots(**args)):
        raise ValueError("R4u output must be outside all original evidence")
    if out.exists() or out.is_symlink():
        raise FileExistsError("R4u source-only output is create-only")
    out.mkdir(parents=True, exist_ok=False)
    try:
        for arm in ARMS:
            folder = out / arm
            folder.mkdir()
            (folder / "input").mkdir()
            for relative in (CHECKPOINT, SCRIPT):
                src = Path(r4t_study) / arm / relative
                dst = folder / relative
                shutil.copyfile(src, dst, follow_symlinks=False)
                if _digest(src) != _digest(dst):
                    raise ValueError("R4u checkpoint or Tcl copy SHA mismatch")
            if (_digest(folder / CHECKPOINT) !=
                    original["arms"][arm]["posthoc_original_checkpoint_sha256"]
                    or (folder / SCRIPT).read_text() != r4t._script()):
                raise ValueError("R4u original routed checkpoint/Tcl not pinned")
        result = {
            "schema": SCHEMA,
            "status": "R4U_ISOLATED_REPORT_ONLY_NATIVE_INPUTS_PREPARED",
            "original_source_revision": original["original_source_revision"],
            "original_R4t_study": str(Path(r4t_study)),
            "original_R4p_study": str(Path(study)),
            "original_checkpoint_provenance": "POST_HOC_SHA_NOT_EXECUTION_TIME_SIGNED",
            "checkpoint_sha256": {
                arm: original["arms"][arm]["posthoc_original_checkpoint_sha256"]
                for arm in ARMS
            },
            "original_routed_wns_ns": {
                arm: original["arms"][arm]["original_routed_wns_ns"]
                for arm in ARMS
            },
            "paired_report_only_cases": 2,
            "actual_native_vivado_calls": 0,
            "strict_physical_E4": None,
            "board_qualified": False,
        }
        (out / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        signatures = {
            str(p.relative_to(out)): _digest(p)
            for p in sorted(out.rglob("*")) if p.is_file()
        }
        (out / "artifact_sha256.json").write_text(
            json.dumps(signatures, indent=2, sort_keys=True) + "\n")
        return result
    except BaseException as exc:
        (out / "FAILED.json").write_text(json.dumps({
            "exception_type": type(exc).__name__, "detail": str(exc)[:2000],
            "original_vendor_artifacts_modified": False,
        }, indent=2, sort_keys=True) + "\n")
        raise


def audit_inputs(*, native_study: str | Path, r4t_study: str | Path,
                 study: str | Path, r4o_study: str | Path,
                 r4k_study: str | Path, original_r4d: str | Path,
                 original_r4a: str | Path) -> dict:
    args = dict(r4t_study=r4t_study, study=study, r4o_study=r4o_study,
                r4k_study=r4k_study, original_r4d=original_r4d,
                original_r4a=original_r4a)
    original = _original_r4t(**args)
    root = Path(native_study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4u native study folder missing/symlinked")
    resolved = root.resolve()
    if any(resolved == p or p in resolved.parents or resolved in p.parents
           for p in _roots(**args)):
        raise ValueError("R4u original/native output tree overlaps")
    summary = _json(root / "summary.json")
    if (summary.get("schema") != SCHEMA
            or summary.get("status") !=
                "R4U_ISOLATED_REPORT_ONLY_NATIVE_INPUTS_PREPARED"
            or summary.get("original_R4t_study") != str(Path(r4t_study))
            or summary.get("original_R4p_study") != str(Path(study))
            or summary.get("original_source_revision") !=
                original["original_source_revision"]
            or summary.get("original_checkpoint_provenance") !=
                "POST_HOC_SHA_NOT_EXECUTION_TIME_SIGNED"
            or summary.get("paired_report_only_cases") != 2
            or summary.get("actual_native_vivado_calls") != 0
            or summary.get("strict_physical_E4") is not None
            or summary.get("board_qualified") is not False):
        raise ValueError("R4u source-only study identity changed")
    expected = {"summary.json"} | {
        f"{arm}/{relative}" for arm in ARMS for relative in
        (CHECKPOINT, SCRIPT)}
    manifest = _json(root / "artifact_sha256.json")
    if set(manifest) != expected or not all(
        _is_sha(d) for d in manifest.values()
    ):
        raise ValueError("R4u staged input SHA inventory incomplete")
    for name, digest in manifest.items():
        if _digest(root / name) != digest:
            raise ValueError("R4u staged source-only SHA changed")
    for arm in ARMS:
        case = root / arm
        inpath = case / "input"
        if (case.is_symlink() or inpath.is_symlink()
                or not case.is_dir() or not inpath.is_dir()
                or {p.name for p in inpath.iterdir()} != {"routed.dcp"}):
            raise ValueError("R4u staged checkpoint path unsafe or unexpected files")
        dcp = case / CHECKPOINT
        if (_digest(dcp) !=
                original["arms"][arm]["posthoc_original_checkpoint_sha256"]
                or _digest(dcp) != _digest(Path(r4t_study) / arm / CHECKPOINT)
                or (case / SCRIPT).read_text() != r4t._script()):
            raise ValueError("R4u staged checkpoint/Tcl not original posthoc input")
    return {
        "schema": SCHEMA,
        "source_gate": "R4P_R4S_R4T_R4U_POSTHOC_COPIED_DCP_SHA_ATTESTED",
        "original_source_revision": original["original_source_revision"],
        "original_checkpoint_provenance":
            "POST_HOC_SHA_NOT_EXECUTION_TIME_SIGNED",
        "planned_report_only_calls": 2,
        "actual_vendor_calls_during_audit": 0,
        "strict_physical_E4": None,
        "board_qualified": False,
    }


def _attempted(case: Path) -> bool:
    return any((case / path).exists() or (case / path).is_symlink()
               for path in ("reports", "logs", "report_only_invocation.json",
                            "vivado.jou", "vivado.log", ".Xil"))


def _stop_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_one(case: Path, executable: str, timeout_seconds: int) -> dict:
    # Called only after exclusive permanent source-revision native lock.
    before = {_rel: _digest(case / _rel)
              for _rel in (CHECKPOINT, SCRIPT)}
    argv = [str(Path(executable).resolve()), "-mode", "batch",
            "-source", SCRIPT, "-notrace"]
    logs = case / "logs"
    logs.mkdir(exist_ok=False)
    stdout = logs / "vivado.stdout.log"
    stderr = logs / "vivado.stderr.log"
    start = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    rc = None
    timed_out = False
    with stdout.open("xb") as out, stderr.open("xb") as err:
        proc = subprocess.Popen(
            argv, cwd=case.resolve(), stdout=out, stderr=err,
            start_new_session=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        print(f"[FPGAI/R4u] report-only Vivado: {case.name} started (pid={proc.pid})",
              flush=True)
        try:
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_group(proc)
            rc = None if timed_out else proc.returncode
        except BaseException:
            _stop_process_group(proc)
            raise
    after = {_rel: _digest(case / _rel)
             for _rel in (CHECKPOINT, SCRIPT)}
    files = {}
    report_dir = case / "reports"
    if report_dir.is_dir() and not report_dir.is_symlink():
        files = {
            name: _digest(report_dir / name, size_limit=MAX_TEXT_BYTES)
            for name in REPORT_NAMES if (report_dir / name).is_file()
        }
    receipt = {
        "schema": SCHEMA + "/one-time-report-only-invocation",
        "executable": argv[0],
        "argv": argv,
        "cwd": str(case.resolve()),
        "started_utc": start,
        "duration_seconds": round(time.monotonic()-started, 3),
        "returncode": rc,
        "timed_out": timed_out,
        "input_sha256_before": before,
        "input_sha256_after": after,
        "input_stable": before == after,
        "stdout_sha256": _digest(stdout),
        "stderr_sha256": _digest(stderr),
        "fresh_report_sha256": files,
        "completed_successfully": (
            rc == 0 and not timed_out and before == after
            and set(files) == set(REPORT_NAMES)),
        "strict_physical_E4": None,
        "board_qualified": False,
    }
    (case / "report_only_invocation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"[FPGAI/R4u] {case.name}: "
          f"{'completed' if receipt['completed_successfully'] else 'incomplete'}; "
          f"exit={rc}; {round(time.monotonic()-started)}s",
          flush=True)
    return receipt


def execute(*, native_study: str | Path, r4t_study: str | Path,
            study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, execute_native: bool = False,
            confirm_report_only_native: bool = False,
            executable: str = "vivado",
            timeout_seconds: int = 3600) -> dict:
    args = dict(native_study=native_study, r4t_study=r4t_study,
                study=study, r4o_study=r4o_study, r4k_study=r4k_study,
                original_r4d=original_r4d, original_r4a=original_r4a)
    base = audit_inputs(**args)
    if not execute_native:
        return {**base, "status": "R4U_DRY_RUN_NO_VENDOR_CALLS",
                "actual_vendor_calls": 0}
    if not confirm_report_only_native:
        raise ValueError("R4u report-only execution requires explicit second confirmation")
    if type(timeout_seconds) is not int or not (1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise ValueError("R4u report-only native execution timeout invalid")
    binary = shutil.which(executable)
    if binary is None or not Path(binary).is_file():
        raise FileNotFoundError("R4u Vivado executable missing")
    root = Path(native_study)
    lock = root / "r4u_native_attempt.json"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError("R4u original one-time native report lock exists; no retries")
    if any(_attempted(root / arm) for arm in ARMS):
        raise FileExistsError("R4u original report-only arm already attempted")
    with lock.open("x", encoding="utf-8") as file:
        file.write(json.dumps({
            "schema": SCHEMA + "/permanent-attempt-lock",
            "source_revision": base["original_source_revision"],
            "arms": list(ARMS), "never_retry_this_study": True,
        }, indent=2, sort_keys=True) + "\n")
    outcomes = []
    for arm in ARMS:
        receipt = _run_one(root / arm, binary, timeout_seconds)
        outcomes.append({
            "arm": arm,
            "exit_code": receipt["returncode"],
            "report_only_process_completed": receipt["completed_successfully"],
        })
        if not receipt["completed_successfully"]:
            break
    return {
        **base,
        "status": "R4U_REPORT_ONLY_ATTEMPTS_RECORDED_AUDIT_REQUIRED",
        "actual_vendor_calls": len(outcomes),
        "outcomes": outcomes,
        "strict_physical_E4": None,
    }


def audit_native(*, native_study: str | Path, r4t_study: str | Path,
                 study: str | Path, r4o_study: str | Path,
                 r4k_study: str | Path, original_r4d: str | Path,
                 original_r4a: str | Path) -> dict:
    base = audit_inputs(
        native_study=native_study, r4t_study=r4t_study, study=study,
        r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    root = Path(native_study)
    lock = root / "r4u_native_attempt.json"
    valid_lock = False
    if lock.exists() or lock.is_symlink():
        data = _json(lock)
        valid_lock = (
            data.get("schema") == SCHEMA + "/permanent-attempt-lock"
            and data.get("source_revision") == base["original_source_revision"]
            and data.get("arms") == list(ARMS)
            and data.get("never_retry_this_study") is True)
    observations = []
    orig = _original_r4t(
        r4t_study=r4t_study, study=study, r4o_study=r4o_study,
        r4k_study=r4k_study, original_r4d=original_r4d,
        original_r4a=original_r4a)
    for arm in ARMS:
        case = root / arm
        detail = {
            "arm": arm,
            "status": "UNKNOWN_OR_INCOMPLETE_REPORT_ONLY_NATIVE_EVIDENCE",
            "original_routed_wns_ns": orig["arms"][arm]["original_routed_wns_ns"],
            "posthoc_checkpoint_sha256":
                orig["arms"][arm]["posthoc_original_checkpoint_sha256"],
            "native_report_receipt_verified": False,
            "new_timing_summary_wns_ns": None,
            "original_to_new_WNS_consistency": None,
            "new_drc_counts_consistent_with_original": None,
            "complete_detailed_path_count": None,
            "sample_detailed_paths": None,
            "strict_physical_E4": None,
            "board_qualified": False,
        }
        receipt_file = case / "report_only_invocation.json"
        if not valid_lock or not receipt_file.is_file() or receipt_file.is_symlink():
            observations.append(detail)
            continue
        receipt = _json(receipt_file)
        expected_inputs = {
            CHECKPOINT: orig["arms"][arm]["posthoc_original_checkpoint_sha256"],
            SCRIPT: _digest(case / SCRIPT),
        }
        argv = receipt.get("argv")
        sha = receipt.get("fresh_report_sha256")
        if not (
            receipt.get("schema") == SCHEMA + "/one-time-report-only-invocation"
            and receipt.get("completed_successfully") is True
            and receipt.get("returncode") == 0
            and receipt.get("timed_out") is False
            and receipt.get("input_stable") is True
            and receipt.get("input_sha256_before") == expected_inputs
            and receipt.get("input_sha256_after") == expected_inputs
            and receipt.get("cwd") == str(case.resolve())
            and isinstance(argv, list) and len(argv) == 6
            and argv[1:] == ["-mode", "batch", "-source", SCRIPT, "-notrace"]
            and receipt.get("executable") == argv[0]
            and type(receipt.get("duration_seconds")) in (int, float)
            and isfinite(receipt["duration_seconds"])
            and receipt["duration_seconds"] >= 0
            and isinstance(sha, dict) and set(sha) == set(REPORT_NAMES)
            and all(_is_sha(value) for value in sha.values())
        ):
            detail["status"] = "REJECTED_NATIVE_RECEIPT_OR_INPUT_PROVENANCE"
            observations.append(detail)
            continue
        log_ok = all(
            _is_sha(receipt.get(field))
            and _digest(case / "logs" / name) == receipt[field]
            for field, name in (
                ("stdout_sha256", "vivado.stdout.log"),
                ("stderr_sha256", "vivado.stderr.log"),
            )
        )
        reports = case / "reports"
        report_ok = (
            reports.is_dir() and not reports.is_symlink()
            and all(_digest(reports / name, size_limit=MAX_TEXT_BYTES) == sha[name]
                    for name in REPORT_NAMES)
        )
        if not (log_ok and report_ok):
            detail["status"] = "REJECTED_NEW_REPORT_OR_LOG_HASH_DRIFT"
            observations.append(detail)
            continue
        detail["native_report_receipt_verified"] = True
        try:
            wns = _worst_negative_slack(
                (reports / "timing_summary.rpt").read_text(errors="replace"))
            drc = _drc_summary(reports / "drc.rpt")
            path_data = r4s.extract_paths(
                (reports / "max_delay_paths.rpt").read_text(errors="replace"))
        except (ValueError, FileNotFoundError) as exc:
            detail["status"] = "UNKNOWN_NEW_REPORT_PARSER_FORMAT"
            detail["parse_problem"] = str(exc)[:400]
            observations.append(detail)
            continue
        original_wns = detail["original_routed_wns_ns"]
        consistent = isfinite(wns) and abs(wns - original_wns) <= 0.005
        original_drc = orig["arms"][arm]["original_signed_drc_report_sha256"]
        # DRC comparison is to original R4s original signed rule COUNTS, not
        # to a hash-equality expectation across newly generated report files.
        old_rules = (r4s.inspect(
            study=study, r4o_study=r4o_study, r4k_study=r4k_study,
            original_r4d=original_r4d,
            original_r4a=original_r4a)["arms"])
        old = next(x for x in old_rules if x["arm"] == arm)
        drc_same = (
            drc.get("summary_verified") is True
            and drc.get("violations") == old["original_drc_total_violations"]
            and {k: v.get("violations") for k, v in drc.get("rules", {}).items()}
                == old["original_drc_rule_counts"]
        )
        detail.update({
            "new_timing_summary_wns_ns": wns,
            "original_to_new_WNS_consistency": consistent,
            "new_drc_counts_consistent_with_original": drc_same,
            "complete_detailed_path_count":
                path_data["sampled_original_path_count"],
            "sample_detailed_paths": path_data["sample_original_paths"],
            "status": (
                "SOURCE_ATTRIBUTED_REPORT_ONLY_PATHS_AVAILABLE_NOT_E4"
                if consistent and drc_same and
                path_data["sampled_original_path_count"] > 0
                else "POSTHOC_CHECKPOINT_REPORT_ONLY_OBSERVATION_INCOMPLETE_OR_DIFFERENT"),
        })
        observations.append(detail)
    return {
        **base,
        "status": "R4U_ORIGINAL_CHECKPOINT_REPORT_ONLY_DIAGNOSTICS_NOT_E4",
        "native_attempt_lock_verified": valid_lock,
        "arms": observations,
        "reported_complete_path_arms": sum(
            x["status"] == "SOURCE_ATTRIBUTED_REPORT_ONLY_PATHS_AVAILABLE_NOT_E4"
            for x in observations),
        "new_vendor_calls_during_audit": 0,
        "original_R4p_vendor_evidence_modified": False,
        "strict_physical_E4": None,
        "board_qualified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4t-study", required=True)
    parser.add_argument("--study", required=True, help="Original R4p study")
    parser.add_argument("--r4o-study", required=True)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--native-study", help="New R4u prepared study")
    parser.add_argument("--prepare-output")
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--confirm-report-only-native", action="store_true")
    parser.add_argument("--audit-native", action="store_true")
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    kwargs = dict(r4t_study=args.r4t_study, study=args.study,
                  r4o_study=args.r4o_study, r4k_study=args.r4k_study,
                  original_r4d=args.original_r4d,
                  original_r4a=args.original_r4a)
    if args.prepare_output:
        if args.execute_native or args.audit_native or args.native_study:
            parser.error("R4u source-only preparation cannot execute or audit")
        value = prepare(**kwargs, output=args.prepare_output)
    else:
        if not args.native_study:
            parser.error("--native-study required unless preparing a new study")
        if args.audit_native:
            if args.execute_native or args.confirm_report_only_native:
                parser.error("R4u read-only audit cannot execute Vivado")
            value = audit_native(native_study=args.native_study, **kwargs)
        else:
            value = execute(
                native_study=args.native_study, **kwargs,
                execute_native=args.execute_native,
                confirm_report_only_native=args.confirm_report_only_native,
                executable=args.vivado, timeout_seconds=args.timeout_seconds)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
