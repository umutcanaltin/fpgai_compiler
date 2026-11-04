"""Read-only full-cohort source-pinned R2i HLS scientific effect table.

All 36 prespecified graph/Phi pairs retained; E2-only undefined latency
remains unknown, never interpreted as a tie. Conditional qualified HLS
effect summaries are explicitly separate from all-case qualification.
No new Vitis, Vivado, GCC, RTL or worktree modification.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean,median

from fpgai_research.experiments.cst_r2i_fanout_native_execute import audit_native

SCHEMA="fpgai-research/cst-R2i-full-original-qualified-native-HLS-effects/v1"
RESOURCE_KEYS=("bram_18k","lut","ff","dsp","uram")


def compute_effects(original:dict)->dict:
    if (original.get("prespecified_source_cases")!=72 or
            original.get("prespecified_matched_pairs")!=36 or
            len(original.get("per_case_status",[]))!=72 or
            len(original.get("all_original_matched_pairs",[]))!=36):
        raise ValueError("R2i effects demand full original 72-case/36-pair cohort")
    cases=original["per_case_status"]
    pairs=original["all_original_matched_pairs"]
    results=[]
    for group,(a,b,p) in enumerate(
        zip(cases[::2],cases[1::2],pairs)
    ):
        if (a["group"]!=group or b["group"]!=group or
                a["case"]!=f"case_{2*group:03d}" or
                b["case"]!=f"case_{2*group+1:03d}" or
                (a["source_storage"],b["source_storage"])!=(
                    "disjoint","per_element_optimal") or
                a["graph"]!=b["graph"] or a["mapping"]!=b["mapping"] or
                p["group"]!=group or
                p["cases"]!=[a["case"],b["case"]]):
            raise ValueError("R2i effects original matched pair identities diverged")
        valid=(
            a["original_CSynth_timing_resources_qualified"] is True
            and b["original_CSynth_timing_resources_qualified"] is True
            and p["both_original_CSynth_timing_resources_qualified"] is True
        )
        csim=(
            a["original_CSim_qualified"] is True
            and b["original_CSim_qualified"] is True
            and p["both_original_CSim_qualified"] is True
        )
        deltas={}
        for key in RESOURCE_KEYS:
            left=(a["hls_estimated_resources"] or {}).get(key)
            right=(b["hls_estimated_resources"] or {}).get(key)
            deltas[key]=(left-right if valid and left is not None
                         and right is not None else None)
            if valid and key!="uram" and deltas[key] is None:
                raise ValueError("R2i E3 accepted but estimated resource missing")
            if valid and deltas[key]!=p[
                "HLS_estimated_resource_delta_disjoint_minus_optimized"][key]:
                raise ValueError("R2i original resource paired delta invalid")
        delta=(a["hls_worst_cycles"]-b["hls_worst_cycles"] if valid else None)
        if valid and delta!=p["HLS_estimated_latency_delta_worst_cycles"]:
            raise ValueError("R2i original HLS latency paired delta invalid")
        if not valid and any(p["HLS_estimated_resource_delta_disjoint_minus_optimized"][
            key] is not None for key in RESOURCE_KEYS):
            raise ValueError("R2i rejected pair has reported HLS resource delta")
        row={
            "group":group,
            "cases":p["cases"],
            "graph":a["graph"],
            "mapping":a["mapping"],
            "both_original_CSim_qualified":csim,
            "both_original_E3_qualified":valid,
            "disjoint_source_scalar_slots":a["source_scratch_unit_slots"],
            "optimized_source_scalar_slots":b["source_scratch_unit_slots"],
            "source_scalar_slot_savings":(
                a["source_scratch_unit_slots"]-b["source_scratch_unit_slots"]),
            "HLS_estimated_BRAM_18K_disjoint":(
                a["hls_estimated_resources"]["bram_18k"] if valid else None),
            "HLS_estimated_BRAM_18K_optimal":(
                b["hls_estimated_resources"]["bram_18k"] if valid else None),
            "HLS_estimated_delta_disjoint_minus_optimal":deltas,
            "HLS_worst_cycles_disjoint":a["hls_worst_cycles"] if valid else None,
            "HLS_worst_cycles_optimal":b["hls_worst_cycles"] if valid else None,
            "HLS_worst_cycle_delta_disjoint_minus_optimal":delta,
            "HLS_worst_cycle_relative_reduction_percent":(
                100*delta/a["hls_worst_cycles"] if valid else None),
            "qualification_status":(
                "BOTH_SOURCE_ATTESTED_E3" if valid
                else "BOTH_E2_ONLY" if csim else "UNKNOWN_OR_FAILED_E2"),
            "disjoint_E3_warning":a.get("warnings") or [],
            "optimal_E3_warning":b.get("warnings") or [],
            "actual_routed_BRAM_or_Fmax_qualified":False,
        }
        if row["source_scalar_slot_savings"]<=0:
            raise ValueError("R2i prepared matched source arm difference is invalid")
        results.append(row)
    qualified=[x for x in results if x["both_original_E3_qualified"]]
    outcomes={
        "reduction":sum(x["HLS_estimated_delta_disjoint_minus_optimal"][
            "bram_18k"]>0 for x in qualified),
        "tie":sum(x["HLS_estimated_delta_disjoint_minus_optimal"][
            "bram_18k"]==0 for x in qualified),
        "regression":sum(x["HLS_estimated_delta_disjoint_minus_optimal"][
            "bram_18k"]<0 for x in qualified),
    }
    if (sum(outcomes.values())!=len(qualified) or
            original["original_CSynth_both_qualified_pairs"]!=len(qualified) or
            original["original_HLS_estimated_BRAM_reductions"]!=outcomes["reduction"] or
            original["original_HLS_estimated_BRAM_ties"]!=outcomes["tie"] or
            original["original_HLS_estimated_BRAM_regressions"]!=outcomes["regression"] or
            original["original_HLS_estimated_BRAM_UNKNOWN"]!=36-len(qualified)):
        raise ValueError("R2i measured cohort effect counts contradict original audit")
    if (original["original_CSim_qualified_cases"]!=sum(
            x["original_CSim_qualified"] for x in cases) or
            original["original_CSynth_timing_resource_qualified_cases"]!=sum(
            x["original_CSynth_timing_resources_qualified"] for x in cases)):
        raise ValueError("R2i source-attested case qualification denominator changed")
    deltas=[x["HLS_worst_cycle_delta_disjoint_minus_optimal"] for x in qualified]
    return {
        "schema":SCHEMA,
        "original_source_revision":original["original_source_revision"],
        "prespecified_original_cases":72,
        "prespecified_original_matched_pairs":36,
        "original_vitis_invocation_receipts_present":
            original["original_vendor_invocation_receipts_present"],
        "original_E2_CSim_qualified_cases":original["original_CSim_qualified_cases"],
        "original_E3_CSynth_qualified_cases":
            original["original_CSynth_timing_resource_qualified_cases"],
        "both_E3_qualified_matched_pairs":len(qualified),
        "not_both_E3_qualified_matched_pairs":36-len(qualified),
        "HLS_estimated_BRAM_reductions_qualified_pairs":outcomes["reduction"],
        "HLS_estimated_BRAM_ties_qualified_pairs":outcomes["tie"],
        "HLS_estimated_BRAM_regressions_qualified_pairs":outcomes["regression"],
        "E3_qualified_HLS_cycle_reductions":sum(x>0 for x in deltas),
        "E3_qualified_HLS_cycle_ties":sum(x==0 for x in deltas),
        "E3_qualified_HLS_cycle_regressions":sum(x<0 for x in deltas),
        "mean_HLS_cycle_reduction_conditional_on_E3":mean(deltas) if deltas else None,
        "median_HLS_cycle_reduction_conditional_on_E3":median(deltas) if deltas else None,
        "HLS_cycle_delta_minimum_conditional_on_E3":min(deltas) if deltas else None,
        "HLS_cycle_delta_maximum_conditional_on_E3":max(deltas) if deltas else None,
        "original_GCC_or_Vitis_or_Vivado_calls_during_effects_export":0,
        "routed_BRAM_or_Fmax_qualified":False,
        "external_compiler_novelty_qualified":False,
        "cases":results,
        "limitations":[
            "Qualified effect estimates conditional on 24 E3 pairs; all 36 retained",
            "12 missing E3 pairs are not BRAM ties, cycle ties, regressions or zeros",
            "Original HLS estimates are not routed timing/BRAM, measured power or deployed bitstream",
            "Tiny, bounded MatMul fanout examples cannot establish general compiler superiority",
        ],
    }


def export(*,study:str|Path,output:str|Path)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2i original read-only effects export must be create-only")
    original=audit_native(study,require_current_source=False)
    report=compute_effects(original)
    root.mkdir(parents=True)
    (root/"effects.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    fields=(
        "group","cases","graph","mapping","both_original_CSim_qualified",
        "both_original_E3_qualified","source_scalar_slot_savings",
        "HLS_estimated_BRAM_18K_disjoint","HLS_estimated_BRAM_18K_optimal",
        "HLS_worst_cycles_disjoint","HLS_worst_cycles_optimal",
        "HLS_worst_cycle_delta_disjoint_minus_optimal",
        "HLS_worst_cycle_relative_reduction_percent",
        "qualification_status",
    )
    with (root/"pairs.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=fields)
        writer.writeheader()
        for item in report["cases"]:
            writer.writerow({k:(json.dumps(item[k]) if k=="cases" else item[k])
                             for k in fields})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    options=parser.parse_args(argv)
    summary=export(study=options.study,output=options.output)
    print(json.dumps({
        key:value for key,value in summary.items() if key!="cases"},
        sort_keys=True,indent=2))


if __name__=="__main__":
    main()
