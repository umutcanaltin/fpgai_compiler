"""E3 effects denominator and missing-metric honesty, synthetic no-vendor controls."""
from __future__ import annotations

from copy import deepcopy

import pytest

from fpgai_research.experiments.cst_r2i_fanout_native_effects import (
    compute_effects,
)

def _original():
    cases=[]
    pairs=[]
    for group in range(36):
        valid=group<24
        before=3 if group<6 else 0
        after=2 if group<6 else 0
        for arm,storage,bram,latency in (
                (0,"disjoint",before,355),
                (1,"per_element_optimal",after,340)):
            idx=2*group+arm
            cases.append({
                "case":f"case_{idx:03d}",
                "group":group,
                "graph":f"graph_{group}",
                "mapping":"scalar",
                "source_storage":storage,
                "source_scratch_unit_slots":30 if arm==0 else 17,
                "original_CSim_qualified":True,
                "original_CSynth_timing_resources_qualified":valid,
                "hls_estimated_resources":(
                    {"bram_18k":bram,"lut":2500-arm*2,
                     "ff":500-arm*2,"dsp":4,"uram":None}
                    if valid else None),
                "hls_worst_cycles":latency if valid else None,
                "warnings":[] if valid else ["original latency_best=undef"],
            })
        pairs.append({
            "group":group,
            "cases":[f"case_{2*group:03d}",f"case_{2*group+1:03d}"],
            "both_original_CSynth_timing_resources_qualified":valid,
            "both_original_CSim_qualified":True,
            "HLS_estimated_resource_delta_disjoint_minus_optimized":{
                "bram_18k":(before-after if valid else None),
                "lut":2 if valid else None,
                "ff":2 if valid else None,
                "dsp":0 if valid else None,
                "uram":None,
            },
            "HLS_estimated_latency_delta_worst_cycles":15 if valid else None,
        })
    return {
        "original_source_revision":"a"*40,
        "prespecified_source_cases":72,
        "prespecified_matched_pairs":36,
        "original_vendor_invocation_receipts_present":72,
        "original_CSim_qualified_cases":72,
        "original_CSynth_timing_resource_qualified_cases":48,
        "original_CSynth_both_qualified_pairs":24,
        "original_HLS_estimated_BRAM_reductions":6,
        "original_HLS_estimated_BRAM_ties":18,
        "original_HLS_estimated_BRAM_regressions":0,
        "original_HLS_estimated_BRAM_UNKNOWN":12,
        "per_case_status":cases,
        "all_original_matched_pairs":pairs,
    }


def test_all_36_prespecified_cases_and_conditional_E3_effects_without_vendor():
    report=compute_effects(_original())
    assert report["prespecified_original_cases"]==72
    assert report["prespecified_original_matched_pairs"]==36
    assert report["original_E2_CSim_qualified_cases"]==72
    assert report["original_E3_CSynth_qualified_cases"]==48
    assert report["both_E3_qualified_matched_pairs"]==24
    assert report["not_both_E3_qualified_matched_pairs"]==12
    assert report["HLS_estimated_BRAM_reductions_qualified_pairs"]==6
    assert report["HLS_estimated_BRAM_ties_qualified_pairs"]==18
    assert report["HLS_estimated_BRAM_regressions_qualified_pairs"]==0
    assert report["E3_qualified_HLS_cycle_reductions"]==24
    assert report["E3_qualified_HLS_cycle_regressions"]==0
    assert report["mean_HLS_cycle_reduction_conditional_on_E3"]==15
    assert report["median_HLS_cycle_reduction_conditional_on_E3"]==15
    assert report["original_GCC_or_Vitis_or_Vivado_calls_during_effects_export"]==0
    assert report["routed_BRAM_or_Fmax_qualified"] is False
    assert len(report["cases"])==36
    assert all(r["HLS_worst_cycle_delta_disjoint_minus_optimal"] is None
               for r in report["cases"][24:])
    assert all(r["qualification_status"]=="BOTH_E2_ONLY"
               for r in report["cases"][24:])
    assert report["cases"][0]["HLS_estimated_delta_disjoint_minus_optimal"][
        "uram"] is None


def test_corrupt_case_and_pair_order_or_resource_delta_is_rejected():
    original=_original()
    original["per_case_status"][1]["case"]="case_999"
    with pytest.raises(ValueError,match="identities"):
        compute_effects(original)
    original=_original()
    original["all_original_matched_pairs"][0][
        "HLS_estimated_resource_delta_disjoint_minus_optimized"]["bram_18k"]=0
    with pytest.raises(ValueError,match="paired delta"):
        compute_effects(original)


def test_inflated_E3_or_imputed_E2_missing_resource_is_rejected():
    original=_original()
    original["original_HLS_estimated_BRAM_reductions"]=8
    with pytest.raises(ValueError,match="contradict"):
        compute_effects(original)
    original=_original()
    original["all_original_matched_pairs"][24][
        "HLS_estimated_resource_delta_disjoint_minus_optimized"]["bram_18k"]=0
    with pytest.raises(ValueError,match="rejected pair"):
        compute_effects(original)


def test_undefined_HLS_latency_cases_remain_12_unknown_not_false_ties():
    original=_original()
    report=compute_effects(original)
    assert sum(x["both_original_E3_qualified"] is False
               for x in report["cases"])==12
    assert all(x["HLS_estimated_BRAM_18K_disjoint"] is None and
               x["HLS_estimated_BRAM_18K_optimal"] is None
               for x in report["cases"][24:])
    assert report["HLS_estimated_BRAM_ties_qualified_pairs"]==18
