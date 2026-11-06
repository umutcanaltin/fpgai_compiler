"""R4q read-only forensic diagnosis of ORIGINAL R4p Vivado OOC reports.

NO native tool execution, NO original file changes and NO relaxation of
R4p's strict E4 gate. Parses signed original route/timing/utilization/DRC
reports to identify *which* required condition blocked physical qualification.
A partial diagnostic is NEVER a new physical E4 result.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from fpgai_research.experiments import cst_r4p_matched_vivado_ooc_native as r4p
from fpgai_research.toolchains.vivado.report import parse_vivado_reports

SCHEMA = "fpgai-research/R4q-original-R4p-read-only-physical-diagnostics/v1"
REPORTS = r4p._REPORTS
NEEDED = ("lut", "ff", "dsp", "bram")
ALL_RESOURCES = ("lut", "ff", "dsp", "bram", "uram")
MAX_REPORT_BYTES = 16 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_VIVADO_ROW = re.compile(
    r"^\s*\|\s*([^|\n]{1,90})\s*\|\s*([^|\n]{0,80})\|", re.M)


def _original_report(case: Path, signed_hashes: dict) -> tuple[Path, dict]:
    folder = case / "vivado" / "reports"
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("R4q signed original report folder missing/unsafe")
    expected = {}
    for name in REPORTS:
        digest = signed_hashes.get(name)
        path = folder / name
        if (not isinstance(digest, str) or not _SHA.fullmatch(digest)
                or path.is_symlink() or not path.is_file()
                or path.stat().st_size == 0
                or path.stat().st_size > MAX_REPORT_BYTES):
            raise ValueError("R4q original report unsigned/missing/unsafe: " + name)
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("R4q original source-attested report SHA drift: " + name)
        expected[name] = actual
    return folder, expected


def _utilization_rows(path: Path) -> list[dict]:
    """Original labeled rows only, NEVER parse these as resource estimates."""
    txt = path.read_text(errors="replace")
    keys = ("lut", "register", "flip flop", "ff", "dsp", "bram",
            "block ram", "uram", "ultraram", "memory", "ramb")
    rows = []
    for match in _VIVADO_ROW.finditer(txt):
        name, value = (x.strip() for x in match.groups())
        if any(token in name.lower() for token in keys):
            rows.append({"source_label": name, "source_value_cell": value})
            if len(rows) == 64:
                break
    return rows


def inspect(*, study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    """Read originals via the *same* provenance gate as published R4p audit."""
    kwargs = dict(study=study, r4o_study=r4o_study,
                  r4k_study=r4k_study, original_r4d=original_r4d,
                  original_r4a=original_r4a)
    prior = r4p.audit_native(**kwargs)
    if (prior.get("schema") != r4p.SCHEMA
            or prior.get("one_time_original_attempt_lock_verified") is not True
            or prior.get("source_Vivado_inputs_sha_attested") is not True
            or prior.get("original_native_invocation_receipts_present") != 2
            or len(prior.get("arms", [])) != 2
            or [arm.get("arm") for arm in prior["arms"]] != list(r4p.ARMS)):
        raise ValueError("R4q original two-arm R4p evidence population not qualified")
    results = []
    for old in prior["arms"]:
        arm = old["arm"]
        result = {
            "arm": arm,
            "original_R4p_verdict": old["status"],
            "original_strict_clean_DRC_E4": old["strict_clean_DRC_E4"],
            "original_source_and_vendor_receipt_attested": (
                old["original_RTL_ROM_stable"] is True
                and old["vendor_completed"] is True),
            "fresh_report_sha256": old["fresh_original_report_sha256"],
            "original_report_sha_verified": False,
            "source_attested_route_complete": None,
            "source_attested_routed_wns_ns": None,
            "source_attested_timing_met": None,
            "source_attested_utilization": {
                key: None for key in ALL_RESOURCES},
            "source_attested_drc_summary": None,
            "unrecognized_or_missing_required_resources": None,
            "sample_original_utilization_rows": None,
            "blocked_conditions": [],
            "physical_E4": None,
            "board_qualified": False,
        }
        if not result["original_source_and_vendor_receipt_attested"]:
            result["blocked_conditions"].append(
                "ORIGINAL_VENDOR_OR_SOURCE_PROVENANCE_NOT_ATTESTED")
            results.append(result)
            continue
        signed = old.get("fresh_original_report_sha256")
        if not isinstance(signed, dict):
            result["blocked_conditions"].append(
                "ORIGINAL_FRESH_ROUTED_REPORT_SHA_UNAVAILABLE")
            results.append(result)
            continue
        native = Path(study) / arm / "vivado"
        folder, hashes = _original_report(Path(study) / arm, signed)
        result["original_report_sha_verified"] = hashes == signed
        if not result["original_report_sha_verified"]:
            raise ValueError("R4q original report SHA set mismatch")
        try:
            details = parse_vivado_reports(native)
        except (ValueError, FileNotFoundError) as exc:
            result["blocked_conditions"].append(
                "PARSE_OR_ROUTE_STATUS_NOT_QUALIFIED")
            result["parser_error"] = type(exc).__name__ + ": " + str(exc)[:500]
            results.append(result)
            continue
        if details.get("raw_report_sha256") != hashes:
            raise ValueError("R4q report changed between provenance check and parse")
        result["source_attested_route_complete"] = (
            details["routed_nets_complete"] is True)
        timing = details["timing"]
        value = timing.get("wns_ns")
        if type(value) in (int, float) and isfinite(value):
            result["source_attested_routed_wns_ns"] = float(value)
        else:
            result["blocked_conditions"].append("WNS_MISSING_OR_NONFINITE")
        result["source_attested_timing_met"] = (
            timing["timing_met"] is True
            if result["source_attested_routed_wns_ns"] is not None
            else None)
        if result["source_attested_timing_met"] is False:
            result["blocked_conditions"].append("NEGATIVE_ROUTED_WNS")
        util = details["utilization"]
        missing = [
            key for key in NEEDED
            if (type(util.get(key)) not in (int, float)
                or not isfinite(util[key]) or util[key] < 0)
        ]
        result["unrecognized_or_missing_required_resources"] = missing
        if missing:
            result["blocked_conditions"].append(
                "MISSING_OR_INVALID_PHYSICAL_UTILIZATION")
            result["sample_original_utilization_rows"] = _utilization_rows(
                folder / "utilization.rpt")
        result["source_attested_utilization"] = {
            key: (
                util.get(key)
                if type(util.get(key)) in (int, float)
                and isfinite(util[key]) and util[key] >= 0
                else None)
            for key in ALL_RESOURCES
        }
        drc = details["drc"]
        result["source_attested_drc_summary"] = drc
        if drc.get("summary_verified") is not True:
            result["blocked_conditions"].append("DRC_SUMMARY_UNVERIFIED")
        elif drc.get("clean") is not True:
            result["blocked_conditions"].append("DRC_NONCLEAN")
        if not result["source_attested_route_complete"]:
            result["blocked_conditions"].append("ROUTE_NOT_COMPLETE")
        # No auto-promotion to physical E4: original R4p verdict is the
        # authoritative publication gate, and R4q only diagnoses it.
        results.append(result)
    return {
        "schema": SCHEMA,
        "status": "SIGNED_ORIGINAL_R4P_POSTRUN_DIAGNOSTICS_NOT_E4",
        "original_R4p_pair_verdict": prior["status"],
        "source_revision": prior["source_revision"],
        "both_original_native_vendor_invocations_attested": (
            all(x["original_source_and_vendor_receipt_attested"] for x in results)),
        "original_physical_E4": prior["physical_E4"],
        "diagnostic_arms": results,
        "report_files_modified": False,
        "new_Vivado_calls": 0,
        "new_Vitis_calls": 0,
        "board_qualified": False,
        "physical_E4": None,
    }


def run(*, study: str | Path, r4o_study: str | Path,
        r4k_study: str | Path, original_r4d: str | Path,
        original_r4a: str | Path, output: str | Path | None = None) -> dict:
    report = inspect(study=study, r4o_study=r4o_study,
                     r4k_study=r4k_study, original_r4d=original_r4d,
                     original_r4a=original_r4a)
    if output is None:
        return report
    dest = Path(output)
    resolved = dest.resolve()
    roots = tuple(Path(p).resolve() for p in (
        study, r4o_study, r4k_study, original_r4d, original_r4a))
    if any(resolved == root or root in resolved.parents
           for root in roots):
        raise ValueError("R4q output must be outside original evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4q diagnostic output is create-only")
    dest.mkdir(parents=True, exist_ok=False)
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (dest / "report.json").write_text(body)
    (dest / "artifact_sha256.json").write_text(json.dumps({
        "report.json": sha256(body.encode()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--r4o-study", required=True)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        study=args.study, r4o_study=args.r4o_study,
        r4k_study=args.r4k_study,
        original_r4d=args.original_r4d,
        original_r4a=args.original_r4a,
        output=args.output,
    ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
