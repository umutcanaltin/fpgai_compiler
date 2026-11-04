"""Read-only HLS scorecard for matched B=1,2,M tensor lifetime schedules.

Distinct from the whole/column/row baseline. Results are absent until
source-attested original Vitis CSim and bounded synthesis exist. This script
never runs HLS and never promotes a source-level scratch ratio to BRAM savings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fpgai_research.experiments.cst_native_campaign import audit


def report(study: str | Path, output: str | Path) -> dict:
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only block sweep scorecard")
    original = audit(study)
    from fpgai_research.experiments.cst_native_campaign import verify_prepared
    manifest, _ = verify_prepared(study, require_current_revision=False)
    if manifest.get("study_family") != "row_block_schedule_v1":
        raise ValueError("Wrong original-HLS study family for row-block report")
    rows = []
    groups = sorted({x["group"] for x in original["cases"]})
    for group in groups:
        members = [x for x in original["cases"] if x["group"] == group]
        if len(members) != 3 or len({x["row_block"] for x in members}) != 3:
            raise ValueError("Missing or duplicated row-block source control")
        members = sorted(members, key=lambda x: x["row_block"])
        if members[0]["row_block"] != 1 or members[1]["row_block"] != 2:
            raise ValueError("Expected one-row and two-row native controls")
        reference = members[0]
        for case in members[1:]:
            qualified = (reference["hls_csim_attested_with_bounded_latency"]
                         and case["hls_csim_attested_with_bounded_latency"])
            base_cycles = reference["latency_worst_cycles"] if qualified else None
            changed_cycles = case["latency_worst_cycles"] if qualified else None
            # Source-level capacity prediction independent of HLS QoR.
            baseline_scratch = reference["declared_scratch_elements"]
            changed_scratch = case["declared_scratch_elements"]
            if changed_scratch != case["row_block"]*baseline_scratch:
                raise ValueError("Typed row-block scratch scale invariant failed")
            rows.append({
                "group": group, "reference_case": reference["case"],
                "variant_case": case["case"],
                "mapping_mode": reference["mapping_mode"],
                "operator_count": case["operator_count"],
                "row_block_reference": 1,
                "row_block_variant": case["row_block"],
                "source_scratch_reference": baseline_scratch,
                "source_scratch_variant": changed_scratch,
                "reference_csim_attested": reference["csim_passed"],
                "variant_csim_attested": case["csim_passed"],
                "original_native_matched_pair_qualified": bool(qualified),
                "reference_hls_worst_cycles": base_cycles,
                "variant_hls_worst_cycles": changed_cycles,
                "hls_cycle_ratio_reference_over_variant": (
                    base_cycles / changed_cycles if qualified and changed_cycles > 0
                    else None),
                "reference_bram_18k": reference["bram_18k"] if qualified else None,
                "variant_bram_18k": case["bram_18k"] if qualified else None,
                "physical_resource_reduction_established": False,
                "postroute_Fmax_established": False,
            })
    summary = {
        "schema": "fpgai-research/cst-block-sweep-native-scorecard/v1",
        "source_revision": original["source_revision"],
        "matched_graphs": len(groups),
        "paired_comparisons": len(rows),
        "pairs_with_original_csim_csynth_measurements": sum(
            x["original_native_matched_pair_qualified"] for x in rows),
        "pairs_without_required_native_measurements": sum(
            not x["original_native_matched_pair_qualified"] for x in rows),
        "tool_invocations_by_report": 0,
        "source_memory_is_not_physical_bram": True,
        "achieved_clock_established": False,
    }
    dest.mkdir(parents=True)
    with (dest/"pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dest/"summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    print(json.dumps(report(a.study, a.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
