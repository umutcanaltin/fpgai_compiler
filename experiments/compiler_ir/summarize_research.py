"""Read-only original CST paper evidence review across distinct experiment cohorts.

The source-only C1/C2/C3 CPU series and the HLS/Vivado v6 physical
study are DIFFERENT Git source revisions and distinct populations.
This review NEVER pools their metrics or calls a warning-only DRC
observation strict E4. Every numeric physical pair is sourced from
the independently audited original physical study, never from a
synthetic source experiment. No native vendor invocation or writes
unless a NEW --output directory is explicitly selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_evidence import collect
from fpgai_research.experiments.cst_nchw_physical import audit_routed_study

SCHEMA="fpgai-research/cst-cross-cohort-read-only-research-review/v1"
_VALID=("QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4",
        "OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS")


def _verified_physical_pairs(physical):
    records=physical.get("cases")
    if (physical.get("prespecified_cases")!=12
            or not isinstance(records,list) or len(records)!=12
            or any(c.get("case")!=f"case_{i:03d}"
                   for i,c in enumerate(records))):
        raise ValueError("Original physical 12-case denominator invalid")
    pairs=[]
    for i in range(0,12,2):
        a,b=records[i:i+2]
        if (a.get("graph")!=b.get("graph")
                or a.get("mapping")!=b.get("mapping")
                or a.get("requested_clock_ns")!=b.get("requested_clock_ns")
                or (a.get("source_storage"),b.get("source_storage"))!=
                   ("real_disjoint","real_shared")):
            raise ValueError("Physical original disjoint/shared pair geometry changed")
        strict=(a.get("routed_qualified") is True
                and b.get("routed_qualified") is True
                and a.get("status")==_VALID[0] and b.get("status")==_VALID[0])
        observed=(a.get("routed_observation_valid") is True
                  and b.get("routed_observation_valid") is True
                  and a.get("status") in _VALID
                  and b.get("status") in _VALID)
        differences=None
        if observed:
            au,bu=a.get("physical_resources"),b.get("physical_resources")
            if not isinstance(au,dict) or not isinstance(bu,dict):
                raise ValueError("Qualified routed pair has no original utilization")
            required=("lut","ff","dsp","bram","uram")
            if any(type(au.get(k)) not in (int,float)
                   or type(bu.get(k)) not in (int,float) for k in required):
                raise ValueError("Incomplete original physical resource vector")
            aw,bw=a.get("routed_wns_ns"),b.get("routed_wns_ns")
            if type(aw) not in (int,float) or type(bw) not in (int,float):
                raise ValueError("Valid pair lacks original WNS")
            differences={
                "disjoint_minus_shared_resources":{
                    k:au[k]-bu[k] for k in required},
                "disjoint_minus_shared_wns_ns":aw-bw,
                "source_case_resource_vectors":{
                    a["case"]:{k:au[k] for k in required},
                    b["case"]:{k:bu[k] for k in required}},
                "source_case_wns_ns":{a["case"]:aw,b["case"]:bw},
                "source_case_drc_rule_counts":{
                    c["case"]:{
                        rule:row.get("violations")
                        for rule,row in (c.get("drc_warning_summary") or {})
                                            .get("rules",{}).items()}
                    for c in (a,b)},
            }
        pairs.append({
            "pair":i//2,"case_ids":[a["case"],b["case"]],
            "status":("STRICT_E4_PAIR" if strict else
                      "EXPLORATORY_ROUTED_PAIR_WITH_WARNINGS" if observed else
                      "UNKNOWN_OR_INCOMPLETE_PAIR"),
            "strict_clean_drc_e4":strict,
            "source_attested_routed_pair":observed,
            "physical_differences":differences,
            "source_hls_revision":physical.get("original_hls_source_revision"),
        })
    return pairs


def summarize_matched_physical_pairs(pairs:list[dict])->dict:
    """Prespecified six-pair descriptive effect counts, never significance.

    Every metric retains the full six-pair denominator, and missing cases
    are UNKNOWN rather than numeric zero. A pair is classified only if
    the independently original-audited physical evidence is valid.
    """
    if (len(pairs)!=6 or
            any(row.get("pair")!=i for i,row in enumerate(pairs))):
        raise ValueError("Summary must preserve the six prespecified physical pairs")
    complete=[p for p in pairs if p["source_attested_routed_pair"]]
    strict=[p for p in pairs if p["strict_clean_drc_e4"]]
    result={
        "prespecified_pairs":6,
        "routed_descriptive_pairs":len(complete),
        "strict_clean_drc_e4_pairs":len(strict),
        "unknown_or_unqualified_pairs":6-len(complete),
        "descriptive_metrics":{},
        "predeclared_effect_sign":"disjoint_minus_shared",
        "inference":"NONE_DESCRIPTIVE_ONLY_NOT_STATISTICAL_OR_CAUSAL_PROOF",
    }
    for name in ("lut","ff","dsp","bram","uram"):
        observed=[pair["physical_differences"][
            "disjoint_minus_shared_resources"][name] for pair in complete]
        if len(observed)!=len(complete) or any(type(v) not in
                (int,float) for v in observed):
            raise ValueError("Original routed resource difference missing")
        result["descriptive_metrics"][name]={
            "shared_uses_less":sum(x>0 for x in observed),
            "equal":sum(x==0 for x in observed),
            "shared_uses_more":sum(x<0 for x in observed),
            "unknown_pairs":6-len(observed),
            "values_disjoint_minus_shared":observed,
            "primary_e4_values_disjoint_minus_shared":[
                pair["physical_differences"][
                    "disjoint_minus_shared_resources"][name] for pair in strict],
        }
    wns=[pair["physical_differences"][
        "disjoint_minus_shared_wns_ns"] for pair in complete]
    result["descriptive_metrics"]["wns_ns"]={
        "shared_has_more_slack":sum(x<0 for x in wns),
        "equal":sum(x==0 for x in wns),
        "shared_has_less_slack":sum(x>0 for x in wns),
        "unknown_pairs":6-len(wns),
        "values_disjoint_minus_shared":wns,
        "primary_e4_values_disjoint_minus_shared":[
            p["physical_differences"]["disjoint_minus_shared_wns_ns"]
            for p in strict],
    }
    return result


def review(*,cpu_series:str|Path,physical_study:str|Path)->dict:
    cpu=collect(cpu_series)
    physical=audit_routed_study(physical_study)
    pairs=_verified_physical_pairs(physical)
    statuses={x["stage"]:x["status"] for x in cpu["stages"]}
    expected=("rl_compatible_ir_r3_r4","two_phi_readers_e1",
              "two_phi_counterfactual_e1","decision_interface",
              "mapped_event_causality","nchw_real_source_rw_e1",
              "graph_semantic_oracle_e0")
    # Preserve original negative outcomes and zero values unchanged.
    metric={x["stage"]:x["original_counters"] for x in cpu["stages"]}
    c123=cpu["three_research_contribution_source_protocols"]
    causal=metric.get("two_phi_counterfactual_e1",{})
    original_causal_complete=(
        statuses.get("two_phi_counterfactual_e1")=="PASSED"
        and causal.get("original_cases")==24
        and causal.get("controlled_phi_contrasts")==18
        and type(causal.get("changed_last_use_contrasts")) is int
        and type(causal.get("changed_scratch_capacity_contrasts")) is int)
    # A successful Phi edit and a different source hash do not prove a
    # changed release or physical storage requirement. Negative
    # controlled effects must remain explicit in the paper audit.
    causality_gate={
        "original_bounded_counterfactual_population_complete":
            original_causal_complete,
        "source_executed_lifetime_difference_observed":(
            causal["changed_last_use_contrasts"]>0
            if original_causal_complete else None),
        "mapping_conditioned_capacity_difference_observed":(
            causal["changed_scratch_capacity_contrasts"]>0
            if original_causal_complete else None),
        "strong_mapping_conditioned_lifetime_claim_qualified":False,
        "novelty_over_prior_art_established":False,
        "interpretation":(
            "NEGATIVE_CAUSAL_RESULT_ZERO_LAST_USE_AND_CAPACITY_EFFECTS"
            if original_causal_complete and
            causal["changed_last_use_contrasts"]==0 and
            causal["changed_scratch_capacity_contrasts"]==0
            else "OUT_OF_SCOPE_OR_PENDING_INDEPENDENT_SEMANTIC_EVIDENCE"),
    }
    return {
        "schema":SCHEMA,
        "source_cohorts":{
            "cpu_compiler_source_revision":cpu["source_revision"],
            "original_native_hls_source_revision":
                physical.get("original_hls_source_revision"),
            "same_git_source_revision":(
                cpu["source_revision"]==
                physical.get("original_hls_source_revision")),
            "cross_revision_performance_comparison_permitted":False,
        },
        "original_cpu":{
            "stage_counts":cpu["stage_counts"],
            "all_40_paper_steps_passed":(
                len(cpu["stages"])==40 and
                cpu["stage_counts"]=={"PASSED":40,"FAILED":0,"BLOCKED":0}),
            "three_contribution_source_protocols":c123,
            "phi_causal_effect_gate":causality_gate,
            "required_stages":{name:{
                "status":statuses.get(name,"MISSING"),
                "original_counters":metric.get(name,{})}
                for name in expected},
            "qualified_source_evidence_not_scientific_novelty":True,
            "rl_learning_advantage_demonstrated":False,
            "trained_RL_policies":0,
        },
        "original_vivado":{
            "prespecified_cases":12,"prespecified_pairs":6,
            "original_receipts":physical.get("vivado_invocation_receipts"),
            "routed_warning_only_cases":physical.get("routed_warning_observation_cases"),
            "valid_observed_cases":physical.get("routed_observation_valid_cases"),
            "strict_clean_drc_e4_cases":physical.get("routed_qualified_cases"),
            "valid_observed_pairs":sum(p["source_attested_routed_pair"] for p in pairs),
            "strict_clean_drc_e4_pairs":sum(p["strict_clean_drc_e4"] for p in pairs),
            "paired_comparisons":pairs,
            "six_pair_prespecified_descriptive_rollup":
                summarize_matched_physical_pairs(pairs),
            "original_vivado_calls_this_review":0,
            "board_bitstream_or_power_claim":False,
        },
        "publication_ready":False,
        "novelty_relative_to_external_prior_art_established":False,
        "warnings":[
            "The complete source-only CPU series and original native HLS cohort have different source revisions; do not pool metrics",
            "Exploratory warning-disclosed routed resources and WNS are NOT strict E4, board readiness or achieved Fmax",
            "Phi owner/source-hash differences alone are not proven lifetime/capacity changes",
            "RL-interface validity is not RL training performance or generalization",
            "External source-pinned peer comparison and real trained network parity remain unexecuted",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-series",required=True)
    parser.add_argument("--physical-study",required=True)
    parser.add_argument("--output",help="Create-only output directory; omit for read-only")
    args=parser.parse_args(argv)
    dest=Path(args.output) if args.output else None
    if dest is not None and (dest.exists() or dest.is_symlink()):
        raise FileExistsError("Original CST research review output must be create-only")
    result=review(cpu_series=args.cpu_series,physical_study=args.physical_study)
    if dest is not None:
        dest.mkdir(parents=True)
        (dest/"cross_cohort_review.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
