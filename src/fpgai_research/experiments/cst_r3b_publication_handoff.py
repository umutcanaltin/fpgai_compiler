"""R3b: read-only publication evidence snapshot and safe R3a native continuation.

No vendor process is launched in status/report mode. Original source and native
case artifacts are immutable. Native execution needs an explicit double opt-in.
Only a separately opted-in, source-attested E2-pass/E3-clock-fail outcome
may be skipped to collect subsequent ORIGINAL cases, never retried/promoted.
All other failed, partial, or unattributed cases block execution.
This is NOT a manuscript acceptance gate or a physical FPGA QoR experiment.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

from fpgai_research.experiments import cst_r3a_conv_dense_native_hls as r3a

SCHEMA = "fpgai-research/R3b-CNN-Dense-publication-handoff/v1"
CASE_NAMES = tuple(f"case_{i:03d}" for i in range(6))
ARTIFACTS = ("vitis_invocation.json", "logs", "fpgai_hls_project")


def snapshot(*, study: str | Path) -> dict:
    """Read-only, source-attributed evidence; UNKNOWN remains UNKNOWN."""
    root = Path(study)
    native = r3a.audit(study=root)
    pairs = r3a.contrasts(study=root)
    rows = native["cases"]
    if len(rows) != len(CASE_NAMES) or tuple(x["case"] for x in rows) != CASE_NAMES:
        raise ValueError("R3b: original six-case population/order mismatch")
    if native["source_revision"] != pairs["source_revision"]:
        raise ValueError("R3b: mixed-source native/pair report revisions")
    if (native["original_source_cases"] != 6
            or native["original_GCC_source_inputs_qualified"] != 12
            or pairs["prespecified_original_neural_pairs"] != 3
            or len(pairs["pairs"]) != 3):
        raise ValueError("R3b: source/GCC/pair denominator mismatch")
    if (sum(bool(x["original_vitis_invocation_present"]) for x in rows)
            != native["original_vitis_invocation_receipts"]
            or sum(bool(x["native_CSim_E2_accepted"]) for x in rows)
            != native["native_CSim_E2_qualified_cases"]
            or sum(bool(x["native_CSynth_E3_qualified"]) for x in rows)
            != native["native_CSynth_E3_qualified_cases"]):
        raise ValueError("R3b: R3a qualification count inconsistent with cases")

    states = []
    for row in rows:
        case = root / row["case"]
        artifacts = [name for name in ARTIFACTS
                     if (case / name).exists() or (case / name).is_symlink()]
        receipt = bool(row["original_vitis_invocation_present"])
        e2 = bool(row["native_CSim_E2_accepted"])
        e3 = bool(row["native_CSynth_E3_qualified"])
        linked_artifacts = [name for name in ARTIFACTS if (case / name).is_symlink()]
        if e3 and (not e2 or not receipt):
            raise ValueError("R3b: E3 without source-attributed E2/receipt")
        if linked_artifacts or (not receipt and artifacts):
            state = "BLOCKED_PARTIAL_OR_UNATTRIBUTED_ATTEMPT"
        elif e3:
            state = "QUALIFIED_HLS_E3_ESTIMATES_ONLY"
        elif receipt:
            state = "ATTEMPTED_BUT_NOT_E3_QUALIFIED"
        else:
            state = "UNATTEMPTED_UNKNOWN"
        states.append({
            "case": row["case"],
            "network": row["network"],
            "functional_sha256": row.get("functional_sha256"),
            "source_cpp_sha256": row.get("source_cpp_sha256"),
            "state": state,
            "E2_original_CSim": e2,
            "E3_original_CSynth": e3,
            "original_vitis_invocation_present": receipt,
            "source_attested_evidence": row.get("source_attested_evidence"),
            "attested_HLS_timing_diagnostic": row.get("attested_HLS_timing_diagnostic"),
            "qualification_status": row["qualification_status"],
            "attempt_artifacts": artifacts,
            "hls_worst_cycles": row["hls_worst_cycles"] if e3 else None,
            "hls_estimated_clock_ns": row["hls_estimated_clock_ns"] if e3 else None,
            "HLS_estimated_resources": row["HLS_estimated_resources"] if e3 else None,
        })
    for pair in pairs["pairs"]:
        a = CASE_NAMES.index(pair["control"])
        b = CASE_NAMES.index(pair["independent_Phi"])
        if pair["both_E3"] != (states[a]["E3_original_CSynth"]
                              and states[b]["E3_original_CSynth"]):
            raise ValueError("R3b: paired E3 denominator disagreement")
    if (sum(bool(p["both_E3"]) for p in pairs["pairs"])
            != pairs["E3_both_qualified_pairs"]):
        raise ValueError("R3b: paired E3 count inconsistency")

    blockers = []
    if native["original_vitis_invocation_receipts"] != 6:
        blockers.append("R3a six-case original Vitis cohort incomplete: "
                        + str(native["original_vitis_invocation_receipts"]) + "/6 attempted")
    original_rejected = [c["case"] for c in states
                         if c["state"] == "ATTEMPTED_BUT_NOT_E3_QUALIFIED"]
    if original_rejected:
        blockers.append("Original attempted E3-rejected case(s) must remain rejected: "
                        + ", ".join(original_rejected))
    if pairs["E3_both_qualified_pairs"] != 3:
        blockers.append("Only " + str(pairs["E3_both_qualified_pairs"])
                        + "/3 source-matched comparisons have both original E3 arms; "
                        "rejected arms are not imputed as ties")
    blockers.extend((
        "No independently verified specific novel mechanism against a competing artifact",
        "No new R3a strict clean-DRC routed E4 comparison or physical QoR evidence",
        "No general fixed-point bit-exact or trained-network accuracy claim",
        "Manuscript/reproducibility artifact and independent scientific review remain required",
    ))
    return {
        "schema": SCHEMA,
        "decision": "MAJOR_REVISION_NOT_SUBMISSION_READY",
        "source_revision": native["source_revision"],
        "source_GCC_E1_qualified_inputs": native["original_GCC_source_inputs_qualified"],
        "original_source_packages": native["original_source_cases"],
        "original_Vitis_invocation_receipts": native["original_vitis_invocation_receipts"],
        "original_CSim_E2_qualified": native["native_CSim_E2_qualified_cases"],
        "original_CSynth_E3_qualified": native["native_CSynth_E3_qualified_cases"],
        "original_native_unattempted": sum(c["state"] == "UNATTEMPTED_UNKNOWN" for c in states),
        "original_native_attempted_not_E3": len(original_rejected),
        "original_attested_timing_failures": sum(_is_attested_clock_failure(c) for c in states),
        "matched_E3_qualified_pairs": pairs["E3_both_qualified_pairs"],
        "matched_E3_unknown_or_rejected_pairs": pairs["E3_unknown_or_rejected_pairs"],
        "physical_E4": None,
        "physical_E4_note": "Not assessed in R3a; older R2y DRC-warned route is not strict E4",
        "cases": states,
        "contrasts": pairs["pairs"],
        "remaining_submission_blockers": blockers,
        "new_original_Vitis_calls_during_snapshot": 0,
        "new_original_Vivado_calls_during_snapshot": 0,
    }


def _is_attested_clock_failure(case: dict) -> bool:
    """Conservatively identify a specific E2-pass and source-attributed E3 clock miss.

    Eligibility to SKIP an old result is not E3 promotion or permission to
    retry it. A generic failed native run / missing receipt is never skippable.
    """
    diagnostic = case.get("attested_HLS_timing_diagnostic")
    if not isinstance(diagnostic, dict):
        return False
    a = diagnostic.get("estimated_clock_ns")
    b = diagnostic.get("effective_budget_ns")
    m = diagnostic.get("margin_ns")
    sha = diagnostic.get("original_HLS_report_sha256")
    return (case.get("state") == "ATTEMPTED_BUT_NOT_E3_QUALIFIED"
            and case.get("qualification_status") == "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
            and case.get("original_vitis_invocation_present") is True
            and case.get("E2_original_CSim") is True
            and case.get("E3_original_CSynth") is False
            and case.get("source_attested_evidence") == "E3-HLS-synthesis-report"
            and set(ARTIFACTS).issubset(set(case.get("attempt_artifacts") or ()))
            and diagnostic.get("feasible") is False
            and all(type(v) in (int, float) and isfinite(v) for v in (a, b, m))
            and a > b > 0 and m < 0
            and isinstance(sha, str) and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha))


def next_case(report: dict, *,
              allow_attested_timing_failure: bool = False) -> int | None:
    """Choose the next ORIGINAL unattempted case; reject all unsafe history."""
    cases = report["cases"]
    if len(cases) != len(CASE_NAMES):
        raise ValueError("R3b: malformed case report")
    for i, case in enumerate(cases):
        if case["case"] != CASE_NAMES[i]:
            raise ValueError("R3b: case order changed")
        if case["state"] == "QUALIFIED_HLS_E3_ESTIMATES_ONLY":
            continue
        if allow_attested_timing_failure and _is_attested_clock_failure(case):
            continue
        if case["state"] != "UNATTEMPTED_UNKNOWN":
            raise RuntimeError(
                "R3b: refusing continuation after attempted/partial/unqualified "
                + case["case"] + "; preserve original and diagnose")
        return i
    return None


def markdown_report(report: dict) -> str:
    """Standalone concise manuscript handoff, not a scientific claim of novelty."""
    lines = [
        "# R3b original CNN/Dense evidence handoff",
        "",
        "Decision: **MAJOR REVISION / NOT SUBMISSION-READY**.",
        "",
        f"Source revision: `{report['source_revision']}`",
        "",
        "| Evidence | Qualified | Denominator |",
        "|---|---:|---:|",
        f"| Source packages | {report['original_source_packages']} | 6 |",
        f"| Original GCC E1 inputs | {report['source_GCC_E1_qualified_inputs']} | 12 |",
        f"| Original native Vitis invocation receipts | {report['original_Vitis_invocation_receipts']} | 6 |",
        f"| Original Vitis CSim E2 | {report['original_CSim_E2_qualified']} | 6 |",
        f"| Original Vitis CSynth E3 | {report['original_CSynth_E3_qualified']} | 6 |",
        f"| Matched two-arm E3 pairs | {report['matched_E3_qualified_pairs']} | 3 |",
        f"| Original completed but rejected E3 | {report['original_native_attempted_not_E3']} | 6 |",
        f"| Original attested effective-clock failures | {report['original_attested_timing_failures']} | 6 |",
        "| R3a strict routed physical E4 | UNKNOWN / not assessed | 3 |",
        "",
        "All resource and cycle metrics below are HLS **estimates**, not routed measurements.",
        "",
        "| Original case | Status | E2 | E3 | HLS BRAM18K estimate |",
        "|---|---|---|---|---:|",
    ]
    for c in report["cases"]:
        resources = c["HLS_estimated_resources"] or {}
        bram = resources.get("bram_18k")
        lines.append(
            f"| {c['case']} | {c['state']} | {c['E2_original_CSim']} | "
            f"{c['E3_original_CSynth']} | {bram if bram is not None else 'UNKNOWN'} |")
    lines += [
        "",
        "Rejected E3 arms keep null HLS resource/cycle comparisons. Source-attested "
        "timing diagnostics (when available) remain descriptive only.",
        "",
        "## Outstanding submission blockers",
        "",
    ]
    lines += ["- " + x for x in report["remaining_submission_blockers"]]
    lines += [
        "",
        "Original R2z/R3a receipts are local, immutable research artifacts. "
        "An unattempted case is never a tie or a zero resource result.",
        "",
    ]
    return "\n".join(lines)


def _create_only_report(report: dict, folder: str | Path) -> None:
    path = Path(folder)
    if path.exists() or path.is_symlink():
        raise FileExistsError("R3b: refusing to overwrite existing report folder")
    path.mkdir(parents=True, exist_ok=False)
    (path / "read_only_evidence.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    (path / "read_only_evidence.md").write_text(markdown_report(report))


def continue_native(*, study: str | Path, max_new_cases: int,
                    execute_native: bool, confirm_isolated_native: bool,
                    vitis_executable: str, timeout_seconds: int,
                    allow_attested_timing_failure: bool = False) -> dict:
    """Foreground one-time execution. Optional skip of attested clock failures only."""
    if execute_native is not True or confirm_isolated_native is not True:
        raise ValueError("R3b native continuation requires BOTH explicit opt-ins")
    if type(max_new_cases) is not int or not 1 <= max_new_cases <= 6:
        raise ValueError("R3b: bounded max-new-cases must be 1..6")
    attempted = []
    for _ in range(max_new_cases):
        before = snapshot(study=study)
        i = next_case(before,
                      allow_attested_timing_failure=allow_attested_timing_failure)
        if i is None:
            break
        r3a.execute(
            study=study, start_case=i, max_cases=1,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable=vitis_executable, timeout_seconds=timeout_seconds)
        attempted.append(CASE_NAMES[i])
        after = snapshot(study=study)
        if (not after["cases"][i]["E3_original_CSynth"]
                and not (allow_attested_timing_failure
                         and _is_attested_clock_failure(after["cases"][i]))):
            raise RuntimeError(
                "R3b: native case " + CASE_NAMES[i]
                + " did not qualify E3. Original preserved; STOP and inspect."
            )
    result = snapshot(study=study)
    result["new_original_Vitis_cases_in_this_command"] = attempted
    result["original_attested_timing_failures_retained"] = [
        c["case"] for c in result["cases"] if _is_attested_clock_failure(c)]
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--continue-native", action="store_true")
    p.add_argument("--execute-native", action="store_true")
    p.add_argument("--continue-past-attested-timing-failure", action="store_true",
                   help="Preserve E2-pass/E3-clock-fail original and try later cases")
    p.add_argument("--confirm-isolated-native", action="store_true")
    p.add_argument("--max-new-cases", type=int, default=1)
    p.add_argument("--vitis-executable", default="vitis_hls")
    p.add_argument("--timeout-seconds", type=int, default=7200)
    p.add_argument("--report-dir", default=None)
    args = p.parse_args(argv)
    if args.continue_native:
        report = continue_native(
            study=args.study, max_new_cases=args.max_new_cases,
            execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            vitis_executable=args.vitis_executable,
            timeout_seconds=args.timeout_seconds,
            allow_attested_timing_failure=args.continue_past_attested_timing_failure)
    else:
        if (args.execute_native or args.confirm_isolated_native
                or args.continue_past_attested_timing_failure):
            p.error("Native flags without --continue-native are not meaningful")
        report = snapshot(study=args.study)
    if args.report_dir is not None:
        _create_only_report(report, args.report_dir)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
