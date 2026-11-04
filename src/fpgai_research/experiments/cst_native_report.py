"""Read-only paired CST original-HLS paper scorecard.

This compares attested Vitis measurements from a frozen source-matched
study. It never invokes tools, fills missing numbers or infers real Fmax.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fpgai_research.experiments.cst_native_campaign import audit


def build_scorecard(study: str | Path, output: str | Path) -> dict:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Create-only CST paper scorecard already exists")
    prior = audit(study)
    source = prior["cases"]
    groups = sorted({x["group"] for x in source})
    rows = []
    complete = 0
    for group in groups:
        items = {x["variant"]: x for x in source if x["group"] == group}
        if set(items) != {"whole_tensor", "column_major", "row_lifetime"}:
            raise ValueError("Incomplete original-HLS matched triplet")
        baseline = items["whole_tensor"]
        for variant in ("column_major", "row_lifetime"):
            current = items[variant]
            measured_pair = bool(
                baseline["hls_csim_attested_with_bounded_latency"]
                and current["hls_csim_attested_with_bounded_latency"])
            if measured_pair:
                base_cycles = baseline["latency_worst_cycles"]
                variant_cycles = current["latency_worst_cycles"]
                # HLS cycle counts from the same target/clock/numeric
                # study, NOT a physical device speedup.
                cycles_ratio = (base_cycles / variant_cycles
                                if variant_cycles > 0 else None)
                delta_cycles = base_cycles - variant_cycles
                complete += 1
            else:
                cycles_ratio = None
                delta_cycles = None
            rows.append({
                "group": group,
                "mapping_mode": baseline["mapping_mode"],
                "reference_case": baseline["case"],
                "variant_case": current["case"],
                "variant": variant,
                "reference_evidence": baseline["native_evidence"],
                "variant_evidence": current["native_evidence"],
                "reference_csim_pass": baseline["csim_passed"],
                "variant_csim_pass": current["csim_passed"],
                "reference_hls_cycles": baseline["latency_worst_cycles"],
                "variant_hls_cycles": current["latency_worst_cycles"],
                "source_attested_bounded_latency_pair": measured_pair,
                "hls_cycle_difference_reference_minus_variant": delta_cycles,
                "hls_report_cycle_ratio_reference_over_variant": cycles_ratio,
                "reference_lut": baseline["lut"],
                "variant_lut": current["lut"],
                "reference_dsp": baseline["dsp"],
                "variant_dsp": current["dsp"],
                "reference_bram_18k": baseline["bram_18k"],
                "variant_bram_18k": current["bram_18k"],
                "clock_target_feasibility_checked": False,
                "postroute_fmax_known": False,
            })
    root.mkdir(parents=True)
    with (root / "paired_hls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": "fpgai-research/cst-paired-hls-paper-report/v1",
        "source_revision": prior["source_revision"],
        "matched_triplet_groups": len(groups),
        "paired_variant_rows": len(rows),
        "paired_rows_with_source_attested_numerical_and_cycles": complete,
        "not_measured_rows": len(rows) - complete,
        "native_tools_executed_by_report": False,
        "physical_speedup_claim_allowed": False,
        "no_native_qualification_by_tool_exit": True,
        "interpretation": (
            "Only original Vitis CSim plus bounded source-attributed CSynth "
            "cycles are shown. No timing-closure, real Fmax, routed resource "
            "or deployment conclusions may be drawn from this table."
        ),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    print(json.dumps(build_scorecard(args.study, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
