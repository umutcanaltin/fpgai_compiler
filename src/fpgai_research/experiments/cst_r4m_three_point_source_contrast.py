"""R4m: source-attested three-point R4e/R4l HLS contrast ledger.

READ ONLY: original global-source full H (R4e), new row-scheduled
full storage (R4l), and original-sha-identical row source (R4e/R4l).

Separates two matched SOURCE interventions:
A -> B: change source producer/reader schedule with full storage.
B -> C: change scratch materialization with row source schedule.
Do NOT call this a randomized factorial effect, proved identical HLS
hardware scheduling, Vivado cell count, physical timing or throughput.
No vendor calls and never touch any original R4a/R4d/R4k evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as r4l

SCHEMA = "fpgai-research/R4m-three-point-source-contrast-HLS-ledger/v1"
RESOURCES = ("bram_18k", "dsp", "lut", "ff", "uram")
METRICS = (*RESOURCES, "worst_cycles", "estimated_clock_period_ns")
SLOTS = ("original_full_global", "new_full_row_schedule",
         "new_row_row_schedule", "original_row_row_schedule")
_DECIMALS = ("estimated_clock_period_ns",)


def _metric_vector(arm: dict) -> dict | None:
    if (arm.get("E2_qualified") is not True
            or arm.get("E3_qualified") is not True):
        return None
    src = arm.get("HLS_estimated_resources")
    if not isinstance(src, dict):
        return None
    result = {}
    for key in RESOURCES:
        val = src.get(key)
        if type(val) is not int or val < 0:
            return None
        result[key] = val
    cycles = arm.get("HLS_worst_cycles")
    clock = arm.get("HLS_estimated_clock_ns")
    if (type(cycles) is not int or cycles < 1
            or type(clock) not in (float, int)
            or not math.isfinite(clock) or clock <= 0):
        return None
    result["worst_cycles"] = cycles
    result["estimated_clock_period_ns"] = float(clock)
    return result


def compare_three(*, original_full: dict, original_row: dict,
                  new_full: dict, new_row: dict,
                  original_source_revision: str,
                  new_source_revision: str) -> dict:
    """Conservative source-controlled comparisons of FOUR receipts/THREE cells."""
    original_src = original_row.get("original_source_sha256")
    new_src = new_row.get("source_sha256")
    if (original_row.get("arm") != "row_H"
            or original_full.get("arm") != "full_H"
            or new_full.get("arm") != "full_storage"
            or new_row.get("arm") != "row_storage"):
        raise ValueError("R4m original/new arm source identities do not match")
    if not isinstance(original_src, str) or original_src != new_src:
        raise ValueError("R4m original row_H must exactly equal new row_storage C++ source")
    for o in (original_full, original_row):
        if o.get("format") != [32, 6]:
            raise ValueError("R4m original fixed numerical format is not 32/6")
    points = {
        "original_full_global": _metric_vector(original_full),
        "new_full_row_schedule": _metric_vector(new_full),
        "new_row_row_schedule": _metric_vector(new_row),
        "original_row_row_schedule": _metric_vector(original_row),
    }
    qualified = all(vector is not None for vector in points.values())
    row_reports_match = (
        qualified
        and points["new_row_row_schedule"] ==
            points["original_row_row_schedule"]
        and new_row.get("HLS_CSynth_report_sha256") is not None
        and new_row["HLS_CSynth_report_sha256"] ==
            original_row.get("HLS_CSynth_report_sha256")
    )
    result = {
        "schema": SCHEMA,
        "original_R4d_source_revision": original_source_revision,
        "R4k_original_source_revision": new_source_revision,
        "common_row_source_sha256": original_src,
        "common_row_HLS_CSynth_xml_sha256": (
            new_row.get("HLS_CSynth_report_sha256")
            if row_reports_match else None),
        "exact_original_row_reproduction_at_E3": bool(row_reports_match),
        "status": "UNKNOWN_UNQUALIFIED_OR_NONREPRODUCED_HLS_LEDGER",
        "E2_E3_qualified_all_four_observations": qualified,
        "points": points,
        "descriptive_source_contrasts": {
            "schedule_change_with_full_H_storage_B_minus_A": None,
            "storage_change_with_row_source_schedule_C_minus_B": None,
            "combined_original_change_D_minus_A": None,
        },
        "additive_path_identity_verified": None,
        "physical_BRAM_delta": None,
        "placed_routed_clock_delta": None,
        "vendor_calls_during_analysis": 0,
        "original_native_evidence_modified": False,
    }
    if not row_reports_match:
        return result

    def delta(a: dict, b: dict) -> dict:
        return {
            key: (round(b[key] - a[key], 9) if key in _DECIMALS
                  else b[key] - a[key])
            for key in METRICS
        }

    d_schedule = delta(points["original_full_global"],
                       points["new_full_row_schedule"])
    d_storage = delta(points["new_full_row_schedule"],
                      points["new_row_row_schedule"])
    d_total = delta(points["original_full_global"],
                    points["original_row_row_schedule"])
    # Path closure is an arithmetic consistency check, not a causal
    # identification proof. We validate even the time-domain float delta.
    closing = all(
        abs(d_schedule[key] + d_storage[key] - d_total[key])
        <= (1e-8 if key in _DECIMALS else 0)
        for key in METRICS
    )
    if not closing:
        raise ValueError("R4m three-point HLS path decomposition does not close")
    result.update({
        "status": "SOURCE_PINNED_THREE_POINT_HLS_DESCRIPTIVE_LEDGER",
        "descriptive_source_contrasts": {
            "schedule_change_with_full_H_storage_B_minus_A": d_schedule,
            "storage_change_with_row_source_schedule_C_minus_B": d_storage,
            "combined_original_change_D_minus_A": d_total,
        },
        "additive_path_identity_verified": True,
    })
    return result


def inspect(*, r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    old = r4e.audit_native(study=original_r4d, original_study=original_r4a)
    new = r4l.audit_native(study=r4k_study,
                           original_r4d=original_r4d,
                           original_r4a=original_r4a)
    if (old.get("source_attested_arms") != 6
            or old.get("prespecified_matched_pairs") != 3
            or new.get("E1_matched_pair_accepted") is not True
            or new.get("paired_HLS_source_arms") != 2
            or new.get("original_R4d_source_revision") != old.get("source_revision")
            or new.get("original_R4a_source_revision") !=
                old.get("original_R4a_source_revision")):
        raise ValueError("R4m original source/audit populations do not match")
    prior = [a for a in old["observations"] if a.get("format") == [32, 6]]
    current = new.get("observations")
    if (len(prior) != 2 or not isinstance(current, list) or len(current) != 2
            or [a.get("arm") for a in prior] != ["full_H", "row_H"]
            or [a.get("arm") for a in current] !=
                ["full_storage", "row_storage"]):
        raise ValueError("R4m original 32/6 matched source arm population incomplete")
    if new.get("package_hashes", {}).get("row_storage", {}).get("model.cpp") != (
            prior[1]["original_source_sha256"]):
        raise ValueError("R4m SHA source identity mismatch across R4e/R4l")
    result = compare_three(
        original_full=prior[0], original_row=prior[1],
        new_full=current[0], new_row=current[1],
        original_source_revision=old["source_revision"],
        new_source_revision=new["source_revision"],
    )
    result["population"] = {
        "original_R4d_predeclared_numeric_format_pairs": 3,
        "original_R4d_E3_qualified_pairs": old.get("E3_qualified_pairs"),
        "R4l_matched_32_6_E3_qualified_pair": new.get("both_E3_qualified"),
    }
    result["limits"] = [
        "Matched SOURCE issue order is not proof of identical scheduled RTL cycles/II.",
        "Only one original 32/6 fanout graph qualifies the HLS comparison.",
        "A three-point path ledger is not a full factorial or randomized experiment.",
        "BRAM and DSP/LUT/FF observations are Vitis HLS estimates, not Vivado physical results.",
        "R4k row source matches original R4e row source byte-for-byte; E3 report matched exactly.",
        "Original R4d 16/6 and 32/8 NEW native arms remain UNKNOWN, not negatives.",
        "Do not infer arbitrary-network compiler generality, FPGA frequency, power or physical area.",
        "Zero new vendor calls; original R4a/R4d/R4k files are never written.",
    ]
    return result


def run(*, r4k_study: str | Path, original_r4d: str | Path,
        original_r4a: str | Path, output: str | Path | None = None) -> dict:
    data = inspect(r4k_study=r4k_study, original_r4d=original_r4d,
                   original_r4a=original_r4a)
    if output is None:
        return data
    target = Path(output)
    dest = target.resolve()
    original_roots = tuple(
        Path(p).resolve() for p in (r4k_study, original_r4d, original_r4a))
    if any(dest == root or root in dest.parents for root in original_roots):
        raise ValueError("R4m report output must be outside original native evidence")
    if target.exists() or target.is_symlink():
        raise FileExistsError("R4m report is create-only")
    target.mkdir(parents=True, exist_ok=False)
    body = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (target / "report.json").write_text(body)
    (target / "artifact_sha256.json").write_text(json.dumps({
        "report.json": sha256(body.encode()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        r4k_study=args.r4k_study,
        original_r4d=args.original_r4d,
        original_r4a=args.original_r4a,
        output=args.output,
    ), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
