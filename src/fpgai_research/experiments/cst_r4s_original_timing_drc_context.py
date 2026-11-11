"""R4s: original R4p Vivado timing-path and DRC-context lexical forensics.

No Vivado/Vitis, no rerun, no report edits. Everything is read from the
original SOURCE/ROM/receipt/report-SHA pinned physical R4q/R4r population.
Detailed paths can be ABSENT from a Vivado timing SUMMARY; do not invent
start/end registers or infer the cause of adverse WNS and LUT/FF deltas.
No physical strict E4 promotion, no board performance claims.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from fpgai_research.experiments import cst_r4q_original_r4p_physical_diagnostics as r4q
from fpgai_research.experiments import cst_r4r_original_routed_tradeoff_ledger as r4r

SCHEMA = "fpgai-research/R4s-original-signed-timing-and-DRC-context/v1"
ARMS = r4r.ARMS
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_PATHS = 5
MAX_CONTEXT_PER_RULE = 3
MAX_TEXT = 260
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_START = re.compile(r"^\s*(?:Startpoint|Source)\s*:\s*(.*?)\s*$", re.I)
_END = re.compile(r"^\s*(?:Endpoint|Destination)\s*:\s*(.*?)\s*$", re.I)
_SLACK = re.compile(
    r"^\s*Slack\s*\(\s*(?:VIOLATED|MET)\s*\)\s*:?\s*("
    + _NUMBER + r")\s*ns\b", re.I)
_GROUP = re.compile(r"^\s*Path\s+Group\s*:\s*(.*?)\s*$", re.I)
_TYPE = re.compile(r"^\s*Path\s+Type\s*:\s*(.*?)\s*$", re.I)


def _text(original: Path, expected_sha: str) -> str:
    if (original.is_symlink() or not original.is_file()
            or original.stat().st_size == 0
            or original.stat().st_size > MAX_REPORT_BYTES):
        raise ValueError("R4s signed original text missing/unsafe")
    data = original.read_bytes()
    if sha256(data).hexdigest() != expected_sha:
        raise ValueError("R4s immutable original report SHA mismatch")
    return data.decode("utf-8", errors="replace")


def _safe_field(value: str) -> str | None:
    value = " ".join(value.strip().split())
    return value[:MAX_TEXT] if value else None


def extract_paths(timing_text: str) -> dict:
    """Only explicit detailed path witnesses, never WNS-only inference."""
    lines = timing_text.splitlines()
    starts = [(i, line) for i, line in enumerate(lines)
              if _START.match(line)]
    found = []
    for index, (start, line) in enumerate(starts):
        end = min(
            start + 85,
            starts[index + 1][0] if index + 1 < len(starts) else len(lines))
        fragment = lines[start:end]
        orig = _safe_field(_START.match(line).group(1))
        endpoint = group = path_type = None
        slack = None
        for ln in fragment:
            m = _END.match(ln)
            if m and endpoint is None:
                endpoint = _safe_field(m.group(1))
            m = _GROUP.match(ln)
            if m and group is None:
                group = _safe_field(m.group(1))
            m = _TYPE.match(ln)
            if m and path_type is None:
                path_type = _safe_field(m.group(1))
            m = _SLACK.match(ln)
            if m:
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                if isfinite(v):
                    slack = v
                    break
        # A source register alone or a path-group summary is NOT a
        # complete path. Do not call a section critical without all three.
        if orig is not None and endpoint is not None and slack is not None:
            found.append({
                "original_source": orig,
                "original_destination": endpoint,
                "original_path_group": group,
                "original_path_type": path_type,
                "path_slack_ns": slack,
                "source_line_number": start + 1,
                "complete_path_detail_in_signed_text": True,
            })
        if len(found) >= MAX_PATHS:
            break
    return {
        "status": (
            "EXPLICIT_ORIGINAL_PATH_DETAILS_PRESENT_NOT_CAUSAL_FORENSICS"
            if found else "UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY"),
        "sampled_original_path_count": len(found),
        "original_startpoint_or_source_headers_seen": len(starts),
        "sample_original_paths": found,
        "path_scope": "At most five source-explicit path sections from the "
                      "existing signed timing summary; not exhaustive.",
        "critical_path_root_cause": None,
    }


def extract_drc_context(drc_text: str, signed_rules: dict) -> dict:
    """Bounded verbatim source examples; not counts, explanations or fixes."""
    lines = drc_text.splitlines()
    output = {}
    for name in sorted(signed_rules):
        matches = []
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
        for index, line in enumerate(lines):
            if not pattern.search(line):
                continue
            clean = _safe_field(line)
            if clean and len(matches) < MAX_CONTEXT_PER_RULE:
                matches.append({
                    "original_report_line_number": index + 1,
                    "original_report_text_excerpt": clean,
                })
        output[name] = {
            "signed_summary_violation_count": signed_rules[name],
            "sample_original_report_lines": matches,
            "status": (
                "EXPLICIT_SOURCE_RULE_LINES_PRESENT_NOT_CAUSAL_EXPLANATION"
                if matches else "UNKNOWN_ORIGINAL_RULE_CONTEXT_ABSENT"),
        }
    return output


def inspect(*, study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    kwargs = dict(study=study, r4o_study=r4o_study,
                  r4k_study=r4k_study, original_r4d=original_r4d,
                  original_r4a=original_r4a)
    original = r4q.inspect(**kwargs)
    ledger = r4r.summarize(original)
    if (ledger["source_attested_original_routed_pair"] is not True
            or ledger["status"] !=
                "SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_TIMING_AND_DRC_NOT_QUALIFIED"
            or ledger["physical_E4"] is not None
            or ledger["original_source_revision"] != r4r.SOURCE_REVISION):
        raise ValueError("R4s original R4p/R4q/R4r source-evidence gate missing")
    arms = []
    for r4q_arm, signed_arm in zip(original["diagnostic_arms"],
                                  ledger["original_arms"]):
        name = signed_arm["arm"]
        if r4q_arm.get("arm") != name or name not in ARMS:
            raise ValueError("R4s original two-arm order corrupted")
        receipt = r4q_arm.get("fresh_report_sha256")
        if not isinstance(receipt, dict) or set(receipt) != set(r4q.REPORTS):
            raise ValueError("R4s original physical report receipt incomplete")
        folder = Path(study) / name / "vivado" / "reports"
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R4s original report folder absent/symlinked")
        timing = _text(folder / "timing_summary.rpt",
                       receipt["timing_summary.rpt"])
        drc = _text(folder / "drc.rpt", receipt["drc.rpt"])
        path_data = extract_paths(timing)
        drc_data = extract_drc_context(drc, signed_arm["drc_rules"])
        # Detect mid-inspection changes; any ambiguity is a hard stop.
        if (sha256((folder / "timing_summary.rpt").read_bytes()).hexdigest() !=
                receipt["timing_summary.rpt"]
                or sha256((folder / "drc.rpt").read_bytes()).hexdigest() !=
                receipt["drc.rpt"]):
            raise ValueError("R4s original evidence changed during read-only inspection")
        arms.append({
            "arm": name,
            "original_signed_timing_summary_sha256": receipt["timing_summary.rpt"],
            "original_signed_drc_report_sha256": receipt["drc.rpt"],
            "original_routed_wns_ns": signed_arm["routed_wns_ns"],
            "original_timing_met": signed_arm["routed_timing_met"],
            "original_drc_total_violations": signed_arm["drc_total_violations"],
            "original_drc_rule_counts": signed_arm["drc_rules"],
            "timing_path_source_details": path_data,
            "drc_rule_source_context": drc_data,
            "routed_timing_failure_cause": None,
            "physical_E4": None,
        })
    return {
        "schema": SCHEMA,
        "status": "SIGNED_ORIGINAL_R4P_TIMING_DRC_TEXT_FORENSICS_NOT_E4",
        "original_source_revision": r4r.SOURCE_REVISION,
        "original_R4r_verdict": ledger["status"],
        "source_attested_arms": 2,
        "arms": arms,
        "source_to_source_WNS_delta_ns": ledger[
            "routed_WNS_delta_row_minus_full_ns"],
        "source_to_source_DRC_warning_delta": ledger[
            "drc_total_delta_row_minus_full"],
        "unavailable_root_cause_claim": (
            "Existing source timing summary and DRC warnings cannot by themselves "
            "establish which RTL logic or physical path caused the LUT/FF/WNS change."),
        "no_new_vivado_or_vitis_calls": True,
        "original_vendor_artifacts_modified": False,
        "physical_E4": None,
        "board_qualified": False,
    }


def run(*, study: str | Path, r4o_study: str | Path,
        r4k_study: str | Path, original_r4d: str | Path,
        original_r4a: str | Path, output: str | Path | None = None) -> dict:
    result = inspect(study=study, r4o_study=r4o_study,
                     r4k_study=r4k_study, original_r4d=original_r4d,
                     original_r4a=original_r4a)
    if output is None:
        return result
    dest = Path(output)
    resolved = dest.resolve()
    roots = [Path(p).resolve() for p in (
        study, r4o_study, r4k_study, original_r4d, original_r4a)]
    if any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError("R4s output must be outside original native evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4s report output is create-only")
    dest.mkdir(parents=True, exist_ok=False)
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (dest / "report.json").write_text(body)
    (dest / "artifact_sha256.json").write_text(json.dumps({
        "report.json": sha256(body.encode()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--r4o-study", required=True)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(study=args.study, r4o_study=args.r4o_study,
                         r4k_study=args.r4k_study,
                         original_r4d=args.original_r4d,
                         original_r4a=args.original_r4a,
                         output=args.output), indent=2, sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
