"""Read-only, source-attributed R2j physical observation diagnostics.

Consume the EXISTING read-only Vivado audit, not fresh vendor runs.
Never promote OOC warning-only observations to strict clean-DRC E4.
Report every original case's DRC rule IDs, routed physical counts,
WNS and original source/RTL provenance; a descriptive matched
difference exists only when both routed observations are valid.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

SOURCE_SCHEMA="fpgai-research/cst-R2j-original-routed-fanout-pair-export/v1"
SCHEMA="fpgai-research/cst-R2j-readonly-physical-diagnostic/v1"
_METRICS=("lut","ff","dsp","bram","uram")


def _read_safe(path:Path):
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size>32*1024*1024):
        raise ValueError("R2j original read-only audit missing/unsafe/oversized")
    return json.loads(path.read_text())


def diagnostic(original_export:str|Path,*,pair:int=20)->dict:
    if type(pair) is not int or not 0<=pair<36:
        raise ValueError("R2j pilot pair must be one of the 36 prespecified pairs")
    root=Path(original_export)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Original read-only R2j export missing")
    manifest=_read_safe(root/"summary.json")
    cases=_read_safe(root/"original_physical_audit.json")
    pairs=_read_safe(root/"matched_pairs.json")
    if (manifest.get("schema")!=SOURCE_SCHEMA
            or manifest.get("prespecified_cases")!=72
            or manifest.get("prespecified_pairs")!=36
            or cases.get("prespecified_cases")!=72
            or type(cases.get("cases")) is not list
            or len(cases["cases"])!=72
            or type(pairs) is not list
            or len(pairs)!=36):
        raise ValueError("R2j pilot diagnostic refuses incomplete original cohort")
    a,b=cases["cases"][2*pair:2*pair+2]
    paired=pairs[pair]
    expected=[f"case_{2*pair:03d}",f"case_{2*pair+1:03d}"]
    if (paired.get("pair")!=pair or paired.get("cases")!=expected
            or [a.get("case"),b.get("case")]!=expected
            or (a.get("source_storage"),b.get("source_storage"))!=(
                "disjoint","per_element_optimal")
            or a.get("graph")!=b.get("graph")
            or a.get("mapping")!=b.get("mapping")
            or a.get("requested_clock_ns")!=b.get("requested_clock_ns")
            or not all(isinstance(x.get("source_cpp_sha256"),str)
                       and len(x["source_cpp_sha256"])==64 for x in (a,b))):
        raise ValueError("R2j pilot pair/source identity mismatch")
    actual_observed=(a.get("routed_observation_valid") is True
                     and b.get("routed_observation_valid") is True)
    actual_strict=(a.get("routed_qualified") is True
                   and b.get("routed_qualified") is True)
    if (paired.get("both_routed_observation_valid") is not actual_observed
            or paired.get("both_source_attributed_routed_timing_drc_qualified")
                  is not actual_strict):
        raise ValueError("R2j claimed pair qualification contradicts original cases")
    observed=actual_observed
    strict=actual_strict
    items=[]
    for item in (a,b):
        drc=item.get("drc_warning_summary") or {}
        resources=item.get("physical_resources") or {}
        if item.get("routed_observation_valid") is True:
            # Vivado's report parser preserves nonintegral resource counts,
            # especially half-BRAM tiles. Accept finite nonnegative numeric
            # values without accepting bools, NaN, inf or missing metrics.
            required=("lut","ff","dsp","bram")
            metrics_ok=all(
                type(resources.get(k)) in (int,float)
                and isfinite(resources[k])
                and resources[k]>=0
                for k in required
            )
            wns=item.get("routed_wns_ns")
            if (item.get("routed_timing_met") is not True
                    or not metrics_ok
                    or type(wns) not in (int,float)
                    or not isfinite(wns)):
                raise ValueError("Claimed routed observation lacks valid timing/resources")
            if not (item.get("routed_qualified") is True
                    or item.get("routed_warning_observation") is True):
                raise ValueError("Routed observation has no approved DRC classification")
        warnings=drc.get("rules") or {}
        items.append({
            "case":item["case"],"source_storage":item["source_storage"],
            "original_source_cpp_sha256":item["source_cpp_sha256"],
            "original_hls_qualified":item.get("original_hls_qualified"),
            "original_vivado_invocation_present":
                item.get("original_vivado_invocation_present"),
            "routed_observation_valid":item.get("routed_observation_valid"),
            "strict_clean_drc_E4":item.get("routed_qualified") is True,
            "routed_warning_observation":
                item.get("routed_warning_observation") is True,
            "routed_WNS_ns":item.get("routed_wns_ns"),
            "source_attributed_physical_resources":{
                k:resources.get(k) if item.get("routed_observation_valid") else None
                for k in _METRICS},
            "DRC_clean_verified":item.get("drc_verified_clean") is True,
            "DRC_summary_verified":drc.get("summary_verified"),
            "DRC_violations":drc.get("violations"),
            "DRC_critical_warnings":drc.get("critical_warnings"),
            "DRC_unconstrained_IO":drc.get("unconstrained_io"),
            "DRC_rule_details":warnings,
            "ROM_source_evidence_mode":item.get("rom_input_evidence_mode"),
        })
    delta={}
    for key in _METRICS:
        x=items[0]["source_attributed_physical_resources"][key]
        y=items[1]["source_attributed_physical_resources"][key]
        delta[key]=(x-y if observed and x is not None and y is not None else None)
    wns_delta=(items[1]["routed_WNS_ns"]-items[0]["routed_WNS_ns"]
               if observed else None)
    if observed and (delta["bram"]!=paired.get("descriptive_bram_tile_difference")
                     or delta["lut"]!=paired.get("descriptive_lut_difference")
                     or delta["ff"]!=paired.get("descriptive_ff_difference")
                     or delta["dsp"]!=paired.get("descriptive_dsp_difference")):
        raise ValueError("R2j physical observation differs from original matched export")
    if strict and (not all(x["DRC_clean_verified"] for x in items)
                   or paired.get("status")!="MATCHED_OOC_ROUTED_RESULTS"):
        raise ValueError("R2j false clean-DRC E4 promotion")
    return {
        "schema":SCHEMA,
        "original_HLS_source_revision":manifest["source_revision"],
        "physical_pair":pair,
        "original_cases":expected,
        "original_graph":a["graph"],
        "original_mapping":a["mapping"],
        "original_physical_matched_pairs_prespecified":36,
        "original_physical_Vivado_invocations_total":
            manifest["original_vivado_invocation_receipts"],
        "both_source_attributed_routed_observations_valid":observed,
        "both_strict_clean_DRC_E4_qualified":strict,
        "cases":items,
        "descriptive_physical_resource_delta_disjoint_minus_optimized":delta,
        "descriptive_WNS_improvement_optimized_minus_disjoint_ns":wns_delta,
        "qualified_physical_BRAM_savings":delta["bram"] if strict else None,
        "qualified_physical_timing_advantage_ns":wns_delta if strict else None,
        "new_original_GCC_Vitis_Vivado_calls":0,
        "evidence":"READ_ONLY_DERIVED_FROM_ATTESTED_ORIGINAL_VIVADO_OOC_REPORT",
        "limitations":[
            "Descriptive routed resource and WNS differences with DRC warnings are not strict E4 improvements",
            "No achieved board frequency, deployable bitstream or measured power",
            "Only pair-selected descriptive endpoint; do not reclassify other 35 pairs",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-study",required=True)
    p.add_argument("--pair",type=int,default=20)
    p.add_argument("--output",help="Optional new create-only evidence JSON file")
    args=p.parse_args(argv)
    target=Path(args.output) if args.output else None
    if target is not None:
        if target.exists() or target.is_symlink():
            raise FileExistsError("Read-only physical diagnostic output must be create-only")
        if not target.parent.is_dir():
            raise ValueError("Output parent directory does not exist")
    result=diagnostic(args.export_study,pair=args.pair)
    if target is not None:
        with target.open("x",encoding="utf-8") as f:
            f.write(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
