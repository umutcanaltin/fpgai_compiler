"""R4r: source-attested, vendor-free descriptive routed OOC tradeoff ledger.

READ ONLY: takes exactly the two ORIGINAL R4q/R4p Vivado report observations.
No Vivado, Vitis, RTL rewrite, original report rewrite, or E4 promotion.

Original timing violations/DRC warnings are observations, not conveniently
discarded cases. Routed RESOURCE COUNTS with negative WNS are descriptive,
NEVER clean-DRC, timing-qualified physical E4. One graph, one precision.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from fpgai_research.experiments import cst_r4q_original_r4p_physical_diagnostics as r4q

SCHEMA = "fpgai-research/R4r-signed-original-routed-OOC-descriptive-ledger/v1"
SOURCE_REVISION = "776e3a7a4b599fd114718f6a03ab73e06be17f5c"
ARMS = ("full_storage", "row_storage")
RESOURCES = ("bram", "dsp", "lut", "ff", "uram")


def _numeric(value, *, whole=False):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        return False
    return not whole or int(value) == value


def _signed_arm(item: dict) -> dict | None:
    """Reject any incomplete report, rather than assuming absent means zero."""
    if (item.get("arm") not in ARMS
            or item.get("original_source_and_vendor_receipt_attested") is not True
            or item.get("original_report_sha_verified") is not True
            or item.get("source_attested_route_complete") is not True
            or item.get("original_strict_clean_DRC_E4") is not False
            or item.get("physical_E4") is not None
            or item.get("unrecognized_or_missing_required_resources") != []):
        return None
    hashes = item.get("fresh_report_sha256")
    if (not isinstance(hashes, dict)
            or set(hashes) != set(r4q.REPORTS)
            or any(not isinstance(h, str) or len(h) != 64
                   or any(c not in "0123456789abcdef" for c in h)
                   for h in hashes.values())):
        return None
    values = item.get("source_attested_utilization")
    if (not isinstance(values, dict) or
            any(not _numeric(values.get(k)) for k in RESOURCES)):
        return None
    wns = item.get("source_attested_routed_wns_ns")
    if type(wns) not in (int, float) or not isfinite(wns):
        return None
    if item.get("source_attested_timing_met") is not (wns >= 0):
        return None
    drc = item.get("source_attested_drc_summary")
    if not isinstance(drc, dict):
        return None
    rules = drc.get("rules")
    if (drc.get("summary_verified") is not True
            or drc.get("available") is not True
            or type(drc.get("violations")) is not int
            or drc["violations"] < 0
            or not isinstance(rules, dict)
            or not rules):
        return None
    by_rule = {}
    for name, rule in rules.items():
        if (not isinstance(name, str) or not name
                or not isinstance(rule, dict)
                or type(rule.get("violations")) is not int
                or rule["violations"] < 0
                or rule.get("severity") not in (
                    "Warning", "Critical Warning", "Error",
                    "Advisory", "Info")):
            return None
        by_rule[name] = rule["violations"]
    if (sum(by_rule.values()) != drc["violations"]
            or drc.get("clean") is not (drc["violations"] == 0)
            or drc.get("critical_warnings") != sum(
                rule["violations"] for rule in rules.values()
                if rule["severity"] in ("Critical Warning", "Error"))
            or drc.get("board_deployment_qualified") is not False):
        return None
    blocked = item.get("blocked_conditions")
    if not isinstance(blocked, list):
        return None
    if ("NEGATIVE_ROUTED_WNS" in blocked) is not (wns < 0):
        return None
    if ("DRC_NONCLEAN" in blocked) is not (drc["violations"] > 0):
        return None
    return {
        "arm": item["arm"],
        "resources": {k: values[k] for k in RESOURCES},
        "routed_wns_ns": float(wns),
        "routed_timing_met": wns >= 0,
        "drc_total_violations": drc["violations"],
        "drc_critical_warnings": drc["critical_warnings"],
        "drc_rules": by_rule,
        "blocked_conditions": sorted(blocked),
        "source_report_sha256": hashes,
        "strict_clean_DRC_E4": False,
        "board_qualified": False,
    }


def summarize(report: dict) -> dict:
    """No numeric pair delta until both ORIGINAL observations are complete."""
    if (report.get("schema") != r4q.SCHEMA
            or report.get("status") !=
                "SIGNED_ORIGINAL_R4P_POSTRUN_DIAGNOSTICS_NOT_E4"
            or report.get("source_revision") != SOURCE_REVISION
            or report.get("both_original_native_vendor_invocations_attested")
                is not True
            or report.get("original_physical_E4") is not None
            or report.get("physical_E4") is not None
            or report.get("report_files_modified") is not False
            or report.get("new_Vivado_calls") != 0
            or report.get("new_Vitis_calls") != 0
            or report.get("board_qualified") is not False):
        raise ValueError("R4r original R4q source/physical provenance mismatch")
    original_arms = report.get("diagnostic_arms")
    if (not isinstance(original_arms, list)
            or [x.get("arm") for x in original_arms] != list(ARMS)):
        raise ValueError("R4r predeclared two-arm population missing")
    parsed = [_signed_arm(x) for x in original_arms]
    result = {
        "schema": SCHEMA,
        "status": "UNKNOWN_INCOMPLETE_SIGNED_ORIGINAL_ROUTED_OOC_PAIR",
        "original_source_revision": SOURCE_REVISION,
        "original_R4p_pair_status": report["original_R4p_pair_verdict"],
        "original_R4q_status": report["status"],
        "source_attested_original_routed_pair": all(p is not None for p in parsed),
        "original_arms": parsed,
        "physical_resource_delta_row_minus_full": None,
        "physical_resource_percent_delta_row_minus_full": None,
        "routed_WNS_delta_row_minus_full_ns": None,
        "drc_rule_deltas_row_minus_full": None,
        "drc_total_delta_row_minus_full": None,
        "strict_clean_DRC_E4": False,
        "physical_E4": None,
        "board_qualified": False,
        "native_vendor_calls_during_analysis": 0,
        "original_source_and_reports_modified": False,
        "scientific_limit": (
            "Only descriptive routed OOC mapping, NOT timing-qualified "
            "clean-DRC E4, Fmax, board throughput or generality."),
    }
    if any(p is None for p in parsed):
        return result
    full, row = parsed
    deltas = {k: row["resources"][k] - full["resources"][k]
              for k in RESOURCES}
    pct = {k: (
        round(100 * deltas[k] / full["resources"][k], 3)
        if full["resources"][k] > 0 else None)
        for k in RESOURCES}
    rule_keys = sorted(set(full["drc_rules"]) | set(row["drc_rules"]))
    drc_delta = {
        k: row["drc_rules"].get(k, 0) - full["drc_rules"].get(k, 0)
        for k in rule_keys
    }
    total_drc = row["drc_total_violations"] - full["drc_total_violations"]
    if sum(drc_delta.values()) != total_drc:
        raise ValueError("R4r DRC rule deltas do not reconcile with totals")
    both_timing_fail = not full["routed_timing_met"] and not row["routed_timing_met"]
    both_nonclean_drc = (
        full["drc_total_violations"] > 0 and
        row["drc_total_violations"] > 0)
    result.update({
        "status": (
            "SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_TIMING_AND_DRC_NOT_QUALIFIED"
            if both_timing_fail and both_nonclean_drc
            else "SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_NOT_STRICT_E4"),
        "physical_resource_delta_row_minus_full": deltas,
        "physical_resource_percent_delta_row_minus_full": pct,
        "routed_WNS_delta_row_minus_full_ns": round(
            row["routed_wns_ns"] - full["routed_wns_ns"], 9),
        "drc_rule_deltas_row_minus_full": drc_delta,
        "drc_total_delta_row_minus_full": total_drc,
    })
    return result


def run(*, study: str | Path, r4o_study: str | Path,
        r4k_study: str | Path, original_r4d: str | Path,
        original_r4a: str | Path, output: str | Path | None = None) -> dict:
    result = summarize(r4q.inspect(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a))
    if output is None:
        return result
    dest = Path(output)
    resolved = dest.resolve()
    roots = [Path(p).resolve() for p in (
        study, r4o_study, r4k_study, original_r4d, original_r4a)]
    if any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError("R4r output must be outside immutable original evidence")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4r output is create-only")
    dest.mkdir(parents=True, exist_ok=False)
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (dest / "report.json").write_text(body)
    (dest / "artifact_sha256.json").write_text(json.dumps({
        "report.json": sha256(body.encode()).hexdigest(),
    }, sort_keys=True, indent=2) + "\n")
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
    print(json.dumps(run(
        study=args.study, r4o_study=args.r4o_study,
        r4k_study=args.r4k_study, original_r4d=args.original_r4d,
        original_r4a=args.original_r4a, output=args.output,
    ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
