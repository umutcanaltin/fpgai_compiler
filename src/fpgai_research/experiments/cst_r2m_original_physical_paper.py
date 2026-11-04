"""R2m: final immutable-paper physical effects export for the prespecified R2i/R2l cohort.

This module only re-audits saved original source, HLS and Vivado receipts.
It never invokes GCC, Vitis, Vivado or a new physical experiment. All 36
original matched pairs stay in the table; unmeasured physical data are null.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median

from fpgai_research.experiments.cst_r2l_physical_completion import (
    ORIGINAL_HLS_POSITIVE_PAIRS, TOTAL_PAIRS, build_completion_ledger,
)

SCHEMA = "fpgai-research/cst-R2m-final-original-six-pair-paper-effects/v1"
CASE_REPORTS = ("route_status.rpt", "timing_summary.rpt", "utilization.rpt", "drc.rpt")
RESOURCE_KEYS = ("bram", "lut", "ff", "dsp")
EXPECTED_FOLLOWUP = (23, 28, 31, 32, 35)


def _routed_arm(case: dict) -> dict:
    if (case.get("routed_observation_valid") is not True
            or case.get("routed_timing_met") is not True):
        raise ValueError("R2m selected arm is not a routed timing observation")
    fingerprints = case.get("original_routed_report_sha256")
    if (not isinstance(fingerprints, dict)
            or any(not isinstance(fingerprints.get(k), str)
                   or len(fingerprints[k]) != 64 for k in CASE_REPORTS)):
        raise ValueError("R2m routed report digest provenance absent")
    wns = case.get("routed_wns_ns")
    if type(wns) not in (int, float) or not isfinite(wns):
        raise ValueError("R2m WNS absent or nonfinite")
    resources = case.get("physical_resources")
    if (not isinstance(resources, dict)
            or any(type(resources.get(k)) not in (int, float)
                   or not isfinite(resources[k]) or resources[k] < 0
                   for k in RESOURCE_KEYS)):
        raise ValueError("R2m original routed resources absent or invalid")
    drc = case.get("drc_warning_summary")
    if (not isinstance(drc, dict) or drc.get("summary_verified") is not True
            or type(drc.get("violations")) is not int):
        raise ValueError("R2m original DRC summary not independently verified")
    rules = drc.get("rules") or {}
    if not isinstance(rules, dict):
        raise ValueError("R2m DRC rules malformed")
    if case.get("routed_qualified") is True:
        if (case.get("drc_verified_clean") is not True
                or drc["violations"] != 0
                or rules):
            raise ValueError("R2m E4 cannot have reported DRC violations")
        tier = "STRICT_CLEAN_DRC_E4"
    elif case.get("routed_warning_observation") is True:
        if (drc["violations"] <= 0 or case.get("drc_verified_clean") is True
                or not rules):
            raise ValueError("R2m warned observation must retain DRC violations")
        tier = "DESCRIPTIVE_WARNING_ONLY"
    else:
        raise ValueError("R2m incomplete DRC evidence for selected route")
    return {
        "case": case["case"],
        "tier": tier,
        "wns_ns": wns,
        "resources": {k: resources[k] for k in RESOURCE_KEYS},
        "drc_violations": drc["violations"],
        "drc_rule_counts": {
            k: (v.get("violations") if isinstance(v, dict) else None)
            for k, v in sorted(rules.items())
        },
        "rom_input_evidence_mode": case.get("rom_input_evidence_mode"),
        "original_report_sha256": {k: fingerprints[k] for k in CASE_REPORTS},
    }


def compile_paper_results(*, native: dict, effects: dict, original: dict,
                          followup: dict) -> dict:
    """Transform independently audited source/physical data into 36 honest rows."""
    ledger = build_completion_ledger(
        native=native, effects=effects, original_physical=original,
        followup_physical=followup,
    )
    if (tuple(ledger["original_physical_observation_pair_indices"]) != (20,)
            or tuple(ledger["separate_followup_observation_pair_indices"])
                != EXPECTED_FOLLOWUP
            or ledger["original_HLS_positive_pairs_with_routed_observations"] != 6
            or ledger["HLS_positive_unobserved_pairs"] != 0):
        raise ValueError("R2m final cohort requires exactly six completed prespecified pairs")
    if len(effects.get("cases", [])) != TOTAL_PAIRS:
        raise ValueError("R2m cannot drop original HLS unknown pairs")
    rows = []
    for i in range(TOTAL_PAIRS):
        r2l = ledger["rows"][i]
        hls = effects["cases"][i]
        positive = i in ORIGINAL_HLS_POSITIVE_PAIRS
        case0, case1 = 2 * i, 2 * i + 1
        arm0 = arm1 = None
        origin = None
        if positive:
            origin = "R2j_original_pilot" if i == 20 else "R2l_separate_followup"
            physical = original if i == 20 else followup
            arm0, arm1 = (_routed_arm(physical["cases"][case0]),
                          _routed_arm(physical["cases"][case1]))
            if (arm0["case"] != r2l["cases"][0]
                    or arm1["case"] != r2l["cases"][1]):
                raise ValueError("R2m selected physical case identity changed")
            selected = r2l["selected_physical_observation"]
            if not selected or not selected["both_routed_observation_valid"]:
                raise ValueError("R2m expected matched routed observation missing")
            for key in RESOURCE_KEYS:
                delta = arm0["resources"][key] - arm1["resources"][key]
                if abs(delta - selected["routed_resource_delta_disjoint_minus_optimal"][key]) > 1e-9:
                    raise ValueError("R2m independent routed resource delta mismatch")
            if abs((arm1["wns_ns"] - arm0["wns_ns"]) -
                   selected["optimized_minus_disjoint_WNS_ns"]) > 1e-9:
                raise ValueError("R2m independently verified WNS delta mismatch")
            if ((arm0["tier"] == "STRICT_CLEAN_DRC_E4"
                 and arm1["tier"] == "STRICT_CLEAN_DRC_E4")
                    is not selected["both_strict_clean_DRC_E4"]):
                raise ValueError("R2m strict pair qualification mismatch")
        else:
            if r2l["selected_physical_observation"] and r2l[
                    "selected_physical_observation"]["both_routed_observation_valid"]:
                raise ValueError("R2m unregistered pair unexpectedly physically observed")
        hls_cycles = hls.get("HLS_worst_cycle_delta_disjoint_minus_optimal")
        row = {
            "pair": i,
            "cases": r2l["cases"],
            "graph": r2l["graph"],
            "mapping": r2l["mapping"],
            "E3_qualified": r2l["both_original_E3_qualified"],
            "source_scratch_unit_slots_saved": hls["source_scalar_slot_savings"],
            "HLS_estimated_BRAM_18K_saved": r2l[
                "HLS_estimated_delta_BRAM_18K_disjoint_minus_optimal"],
            "HLS_worst_cycle_reduction": hls_cycles if r2l[
                "both_original_E3_qualified"] else None,
            "prespecified_HLS_BRAM_positive": positive,
            "physical_study_origin": origin,
            "physical_observation_valid": positive,
            "strict_clean_DRC_E4_pair": (
                arm0["tier"] == arm1["tier"] == "STRICT_CLEAN_DRC_E4"
                if positive else False),
            "routed_disjoint": arm0,
            "routed_optimized": arm1,
            "physical_BRAM_tiles_saved_descriptive": (
                arm0["resources"]["bram"] - arm1["resources"]["bram"]
                if positive else None),
            "physical_LUT_saved_descriptive": (
                arm0["resources"]["lut"] - arm1["resources"]["lut"]
                if positive else None),
            "physical_FF_saved_descriptive": (
                arm0["resources"]["ff"] - arm1["resources"]["ff"]
                if positive else None),
            "physical_DSP_saved_descriptive": (
                arm0["resources"]["dsp"] - arm1["resources"]["dsp"]
                if positive else None),
            "optimized_minus_disjoint_WNS_ns_descriptive": (
                arm1["wns_ns"] - arm0["wns_ns"] if positive else None),
        }
        if positive and (row["HLS_estimated_BRAM_18K_saved"] is None
                         or row["HLS_estimated_BRAM_18K_saved"] <= 0):
            raise ValueError("R2m prespecified HLS-positive pair lost source effect")
        if not positive and row["physical_BRAM_tiles_saved_descriptive"] is not None:
            raise ValueError("R2m physical unknown must never be imputed as a tie")
        rows.append(row)
    observed = [r for r in rows if r["physical_observation_valid"]]
    wns = [r["optimized_minus_disjoint_WNS_ns_descriptive"] for r in observed]
    savings = [r["physical_BRAM_tiles_saved_descriptive"] for r in observed]
    # Positive optimized-minus-disjoint values indicate *higher* physical
    # use. These are descriptive measurements on six selected warned routes.
    lut_changes = [-r["physical_LUT_saved_descriptive"] for r in observed]
    ff_changes = [-r["physical_FF_saved_descriptive"] for r in observed]
    strict = sum(r["strict_clean_DRC_E4_pair"] for r in observed)
    return {
        "schema": SCHEMA,
        "original_source_revision": ledger["original_HLS_source_revision"],
        "original_prespecified_cases": 72,
        "original_prespecified_pairs": 36,
        "both_original_E3_pairs": ledger["original_both_E3_qualified_pairs"],
        "E3_unknown_pairs": ledger["original_E3_unknown_pairs"],
        "HLS_positive_pair_indices": list(ORIGINAL_HLS_POSITIVE_PAIRS),
        "HLS_positive_pairs_count": len(ORIGINAL_HLS_POSITIVE_PAIRS),
        "selected_original_routed_pairs": len(observed),
        "remaining_selected_positive_pairs_unobserved": 6 - len(observed),
        "total_unobserved_physical_matched_pairs": 36 - len(observed),
        "physical_descriptive_BRAM_saving_pairs": sum(x > 0 for x in savings),
        "physical_descriptive_BRAM_tie_pairs": sum(x == 0 for x in savings),
        "physical_descriptive_BRAM_regression_pairs": sum(x < 0 for x in savings),
        "strict_clean_DRC_E4_qualified_pairs": strict,
        "warning_only_descriptive_pairs": len(observed) - strict,
        "descriptive_optimized_WNS_increase_pairs": sum(x > 0 for x in wns),
        "descriptive_optimized_WNS_tie_pairs": sum(x == 0 for x in wns),
        "descriptive_optimized_WNS_decrease_pairs": sum(x < 0 for x in wns),
        "descriptive_mean_optimized_minus_disjoint_WNS_ns": mean(wns),
        "descriptive_median_optimized_minus_disjoint_WNS_ns": median(wns),
        "descriptive_optimized_LUT_increase_pairs": sum(x > 0 for x in lut_changes),
        "descriptive_optimized_LUT_decrease_pairs": sum(x < 0 for x in lut_changes),
        "descriptive_optimized_LUT_tie_pairs": sum(x == 0 for x in lut_changes),
        "descriptive_mean_optimized_minus_disjoint_LUT": mean(lut_changes),
        "descriptive_optimized_FF_increase_pairs": sum(x > 0 for x in ff_changes),
        "descriptive_optimized_FF_decrease_pairs": sum(x < 0 for x in ff_changes),
        "descriptive_optimized_FF_tie_pairs": sum(x == 0 for x in ff_changes),
        "descriptive_mean_optimized_minus_disjoint_FF": mean(ff_changes),
        "new_original_GCC_Vitis_Vivado_calls_during_export": 0,
        "note_prior_foreground_followup_Vivado_calls": 10,
        "board_bitstream_qualified": False,
        "achieved_Fmax_qualified": False,
        "external_baseline_parity_qualified": False,
        "rows": rows,
        "limitations": [
            "Six preselected HLS-positive memory pairs only, not a random or universal FPGA sample",
            "Six warning-only OOC routed BRAM ties are descriptive, not clean-DRC E4 results",
            "30 other original physical pairs have unknown physical outcomes, not BRAM ties",
            "Routed utilization BRAM tiles, HLS BRAM_18K estimates and source scalar slots are different quantities",
            "Descriptive WNS at requested clock does not prove maximum achieved Fmax",
            "Original artifacts source-attested; no new GCC/Vitis/Vivado during export",
        ],
    }


def _md(report: dict) -> str:
    lines = [
        "# R2m: source-attributed complete positive-pair physical report",
        "",
        f"Original HLS source revision: `{report['original_source_revision']}`.",
        "",
        f"Full original denominator: **72 cases / 36 pairs**; "
        f"**{report['both_original_E3_pairs']}/36** qualified HLS E3 pairs, "
        f"**{report['E3_unknown_pairs']}/36** E3 unknown.",
        f"Prespecified HLS-estimated BRAM-positive pairs: "
        f"**{report['HLS_positive_pairs_count']}/36**; routed observations: "
        f"**{report['selected_original_routed_pairs']}/6**, "
        f"**{report['physical_descriptive_BRAM_tie_pairs']}/6 routed BRAM ties**, "
        f"**{report['physical_descriptive_BRAM_saving_pairs']}/6 reductions**, "
        f"**{report['strict_clean_DRC_E4_qualified_pairs']}/6 strict clean-DRC E4**.",
        "",
        "| Pair | Original cases | HLS BRAM_18K saved (estimate) | Routed BRAM tiles disjoint → optimized | "
        "LUT disjoint → optimized | FF disjoint → optimized | DSP disjoint → optimized | WNS ns disjoint → optimized | DRC tier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        if not row["prespecified_HLS_BRAM_positive"]:
            continue
        a, b = row["routed_disjoint"], row["routed_optimized"]
        tier = ("strict E4" if row["strict_clean_DRC_E4_pair"] else "warning-only")
        lines.append(
            f"| {row['pair']} | {a['case']}/{b['case']} | "
            f"{row['HLS_estimated_BRAM_18K_saved']} | "
            f"{a['resources']['bram']:g} → {b['resources']['bram']:g} | "
            f"{a['resources']['lut']:g} → {b['resources']['lut']:g} | "
            f"{a['resources']['ff']:g} → {b['resources']['ff']:g} | "
            f"{a['resources']['dsp']:g} → {b['resources']['dsp']:g} | "
            f"{a['wns_ns']:+.3f} → {b['wns_ns']:+.3f} | {tier} |"
        )
    lines.extend([
        "",
        f"Descriptive optimized-minus-disjoint WNS: "
        f"{report['descriptive_optimized_WNS_increase_pairs']} positive, "
        f"{report['descriptive_optimized_WNS_decrease_pairs']} negative; "
        f"mean {report['descriptive_mean_optimized_minus_disjoint_WNS_ns']:+.6f} ns.",
        "",
        f"Descriptive routed LUT changes (optimized minus disjoint): "
        f"{report['descriptive_optimized_LUT_increase_pairs']} increases, "
        f"{report['descriptive_optimized_LUT_decrease_pairs']} decreases; "
        f"mean {report['descriptive_mean_optimized_minus_disjoint_LUT']:+.3f} LUT.",
        f"Descriptive routed FF changes (optimized minus disjoint): "
        f"{report['descriptive_optimized_FF_increase_pairs']} increases, "
        f"{report['descriptive_optimized_FF_decrease_pairs']} decreases; "
        f"mean {report['descriptive_mean_optimized_minus_disjoint_FF']:+.3f} FF.",
        "",
        "**Interpretation:** The six original HLS-estimated BRAM savings did not translate into "
        "six routed BRAM tile reductions; all six measured pairs tied. None of these "
        "warning-only pairs qualifies as strict clean-DRC E4. The 30 other matched "
        "physical pairs remain unmeasured/unknown; source and HLS effects retain "
        "their distinct evidence tiers. Do not claim board Fmax, power or compiler "
        "firstness from this result.",
        "",
        "Every selected original arm's four Vivado report SHA-256 fingerprints, DRC rule "
        "counts and individual physical resources are in `results.json`; the complete "
        "36-pair table with null physical values for unmeasured cases is in "
        "`all_36_pairs.csv`.",
        "",
    ])
    return "\n".join(lines)


def export(*, hls_study: str | Path, original_physical_study: str | Path,
           followup_study: str | Path, output: str | Path) -> dict:
    out = Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("R2m final original physical export must be create-only")
    from fpgai_research.experiments.cst_r2i_fanout_native_execute import audit_native
    from fpgai_research.experiments.cst_r2i_fanout_native_effects import compute_effects
    from fpgai_research.experiments.cst_r2j_fanout_physical import audit_routed_study

    original_native = audit_native(hls_study, require_current_source=False)
    effects = compute_effects(original_native)
    pilot = audit_routed_study(original_physical_study)
    followup = audit_routed_study(followup_study)
    report = compile_paper_results(
        native=original_native, effects=effects, original=pilot, followup=followup,
    )
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = (
        "pair", "cases", "graph", "mapping", "E3_qualified",
        "source_scratch_unit_slots_saved", "HLS_estimated_BRAM_18K_saved",
        "HLS_worst_cycle_reduction", "prespecified_HLS_BRAM_positive",
        "physical_study_origin", "physical_observation_valid", "strict_clean_DRC_E4_pair",
        "physical_BRAM_tiles_saved_descriptive", "physical_LUT_saved_descriptive",
        "physical_FF_saved_descriptive", "physical_DSP_saved_descriptive",
        "optimized_minus_disjoint_WNS_ns_descriptive",
    )
    with (out / "all_36_pairs.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({
                k: json.dumps(row[k]) if k == "cases" else row[k] for k in fields
            })
    (out / "paper_table.md").write_text(_md(report))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hls-study", required=True)
    parser.add_argument("--original-physical-study", required=True)
    parser.add_argument("--followup-study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = export(
        hls_study=args.hls_study,
        original_physical_study=args.original_physical_study,
        followup_study=args.followup_study,
        output=args.output,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
