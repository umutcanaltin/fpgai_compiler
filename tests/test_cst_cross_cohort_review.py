"""Synthetic cross-cohort summaries never stand in for original Vivado tools."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _sut(monkeypatch):
    folder=Path(__file__).resolve().parents[1]/"experiments"/"compiler_ir"
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location(
        "cst_cross_cohort_review",folder/"summarize_research.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def _fixture(*,strict=False):
    stages=[]
    for name in ("decision_interface","rl_compatible_ir_r3_r4",
                 "two_phi_readers_e1","two_phi_counterfactual_e1",
                 "mapped_event_causality","nchw_real_source_rw_e1",
                 "graph_semantic_oracle_e0"):
        stages.append({
            "stage":name,"status":"PASSED",
            "original_counters":{"negative_interventions":0,
                                 "changed_last_use_contrasts":0}})
    for i in range(40-len(stages)):
        stages.append({"stage":"other_"+str(i),"status":"PASSED",
                       "original_counters":{}})
    cpu={
        "source_revision":"c"*40,
        "stages":stages,
        "stage_counts":{"PASSED":40,"FAILED":0,"BLOCKED":0},
        "three_research_contribution_source_protocols":{
            name:{"source_evidence_complete":True,"novelty_established":False}
            for name in ("C1","C2","C3")},
    }
    cases=[]
    for i in range(12):
        is_pair0=i<2
        valid=is_pair0
        row={
            "case":f"case_{i:03d}",
            "graph":"graph_"+str(i//2),
            "mapping":"mapped_"+str(i//2),
            "source_storage":"real_disjoint" if i%2==0 else "real_shared",
            "requested_clock_ns":32.0,
            "routed_qualified":bool(valid and strict),
            "routed_observation_valid":valid,
            "status":("QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4"
                      if strict and valid else
                      "OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
                      if valid else "PREPARED_NOT_RUN"),
            "physical_resources":(
                {"lut":100 if i%2==0 else 90,
                 "ff":201 if i%2==0 else 200,
                 "dsp":8,"bram":2.5 if i%2==0 else 2.0,"uram":0}
                if valid else None),
            "routed_wns_ns":0.5 if i%2==0 and valid else
                            0.4 if valid else None,
            "drc_warning_summary":(
                {"rules":{"DPIP-2":{"violations":2}}}
                if valid and not strict else
                {"rules":{}} if valid else None),
        }
        cases.append(row)
    physical={
        "prespecified_cases":12,"cases":cases,
        "original_hls_source_revision":"a"*40,
        "routed_warning_observation_cases":0 if strict else 2,
        "routed_observation_valid_cases":2,
        "routed_qualified_cases":2 if strict else 0,
        "vivado_invocation_receipts":2,
    }
    return cpu,physical


def test_review_keeps_warning_only_exploratory_and_reports_real_differences(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    result=sut.review(cpu_series="cpu",physical_study="physical")
    assert result["original_cpu"]["all_40_paper_steps_passed"] is True
    assert result["original_vivado"]["original_receipts"]==2
    assert result["original_vivado"]["valid_observed_pairs"]==1
    assert result["original_vivado"]["strict_clean_drc_e4_pairs"]==0
    first=result["original_vivado"]["paired_comparisons"][0]
    assert first["status"]=="EXPLORATORY_ROUTED_PAIR_WITH_WARNINGS"
    assert first["physical_differences"]["disjoint_minus_shared_resources"]["bram"]==0.5
    assert first["physical_differences"]["source_case_drc_rule_counts"][
        "case_000"]["DPIP-2"]==2
    assert all(x["physical_differences"] is None for x in
               result["original_vivado"]["paired_comparisons"][1:])
    assert result["source_cohorts"]["same_git_source_revision"] is False
    assert result["source_cohorts"]["cross_revision_performance_comparison_permitted"] is False
    assert result["publication_ready"] is False


def test_review_can_report_strict_e4_but_not_claim_publication(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture(strict=True)
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    r=sut.review(cpu_series="cpu",physical_study="phys")
    assert r["original_vivado"]["strict_clean_drc_e4_pairs"]==1
    assert r["original_vivado"]["paired_comparisons"][0]["status"]=="STRICT_E4_PAIR"
    assert r["publication_ready"] is False
    assert r["original_cpu"]["trained_RL_policies"]==0


def test_read_only_review_and_create_only_report(monkeypatch,tmp_path,capsys):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    out=tmp_path/"report"
    sut.main(["--cpu-series","cpu","--physical-study","phys"])
    assert not out.exists()
    printed=json.loads(capsys.readouterr().out)
    assert printed["original_vivado"]["valid_observed_pairs"]==1
    sut.main(["--cpu-series","cpu","--physical-study","phys",
              "--output",str(out)])
    assert (out/"cross_cohort_review.json").is_file()
    with pytest.raises(FileExistsError,match="create-only"):
        sut.main(["--cpu-series","cpu","--physical-study","phys",
                  "--output",str(out)])


@pytest.mark.parametrize("mutation",[
    "missing_case","misordered_case","unmatched_graph",
    "missing_resources","missing_wns",
])
def test_review_rejects_incomplete_or_unattributed_matched_pair(
    monkeypatch,mutation
):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    if mutation=="missing_case":
        phys["cases"].pop()
    elif mutation=="misordered_case":
        phys["cases"][0]["case"]="case_007"
    elif mutation=="unmatched_graph":
        phys["cases"][1]["graph"]="wrong"
    elif mutation=="missing_resources":
        phys["cases"][1]["physical_resources"]["bram"]=None
    elif mutation=="missing_wns":
        phys["cases"][0]["routed_wns_ns"]=None
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    with pytest.raises(ValueError):
        sut.review(cpu_series="cpu",physical_study="phys")


def test_six_pair_rollup_keeps_five_unknown_and_warning_only_denominator(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    result=sut.review(cpu_series="cpu",physical_study="physical")
    rollup=result["original_vivado"]["six_pair_prespecified_descriptive_rollup"]
    assert rollup["prespecified_pairs"]==6
    assert rollup["routed_descriptive_pairs"]==1
    assert rollup["unknown_or_unqualified_pairs"]==5
    assert rollup["strict_clean_drc_e4_pairs"]==0
    assert rollup["descriptive_metrics"]["bram"]["shared_uses_less"]==1
    assert rollup["descriptive_metrics"]["bram"]["unknown_pairs"]==5
    assert rollup["descriptive_metrics"]["bram"][
        "primary_e4_values_disjoint_minus_shared"]==[]
    assert rollup["descriptive_metrics"]["wns_ns"]["shared_has_less_slack"]==1
    assert "NOT_STATISTICAL_OR_CAUSAL_PROOF" in rollup["inference"]


def test_six_pair_rollup_handles_full_cohort_without_cherry_picking(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    for i in range(2,12):
        case=phys["cases"][i]
        case["routed_observation_valid"]=True
        case["status"]="OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
        case["physical_resources"]={
            "lut":100 if i%2==0 else 100,
            "ff":201 if i%2==0 else 200,
            "dsp":8,"bram":2.5,"uram":0,
        }
        case["routed_wns_ns"]=0.5 if i%2==0 else 0.4
        case["drc_warning_summary"]={"rules":{"DPIP-2":{"violations":2}}}
    phys["routed_warning_observation_cases"]=12
    phys["routed_observation_valid_cases"]=12
    phys["vivado_invocation_receipts"]=12
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    result=sut.review(cpu_series="cpu",physical_study="physical")
    rollup=result["original_vivado"]["six_pair_prespecified_descriptive_rollup"]
    assert rollup["routed_descriptive_pairs"]==6
    assert rollup["unknown_or_unqualified_pairs"]==0
    assert rollup["strict_clean_drc_e4_pairs"]==0
    assert rollup["descriptive_metrics"]["bram"]["equal"]==5
    assert rollup["descriptive_metrics"]["bram"]["shared_uses_less"]==1
    assert rollup["descriptive_metrics"]["bram"]["shared_uses_more"]==0
    assert len(rollup["descriptive_metrics"]["bram"][
        "values_disjoint_minus_shared"])==6


def test_zero_phi_lifetime_effect_must_be_explicit_negative_finding(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    for row in cpu["stages"]:
        if row["stage"]=="two_phi_counterfactual_e1":
            row["original_counters"]={
                "original_cases":24,"controlled_phi_contrasts":18,
                "changed_last_use_contrasts":0,
                "changed_scratch_capacity_contrasts":0}
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    result=sut.review(cpu_series="cpu",physical_study="physical")
    gate=result["original_cpu"]["phi_causal_effect_gate"]
    assert gate["original_bounded_counterfactual_population_complete"] is True
    assert gate["source_executed_lifetime_difference_observed"] is False
    assert gate["mapping_conditioned_capacity_difference_observed"] is False
    assert gate["interpretation"]=="NEGATIVE_CAUSAL_RESULT_ZERO_LAST_USE_AND_CAPACITY_EFFECTS"
    assert gate["strong_mapping_conditioned_lifetime_claim_qualified"] is False
    assert gate["novelty_over_prior_art_established"] is False


def test_nonzero_phi_effect_does_not_prove_priority_or_compiler_correctness(monkeypatch):
    sut=_sut(monkeypatch)
    cpu,phys=_fixture()
    for row in cpu["stages"]:
        if row["stage"]=="two_phi_counterfactual_e1":
            row["original_counters"]={
                "original_cases":24,"controlled_phi_contrasts":18,
                "changed_last_use_contrasts":1,
                "changed_scratch_capacity_contrasts":0}
    monkeypatch.setattr(sut,"collect",lambda _:cpu)
    monkeypatch.setattr(sut,"audit_routed_study",lambda _:phys)
    result=sut.review(cpu_series="cpu",physical_study="physical")
    gate=result["original_cpu"]["phi_causal_effect_gate"]
    assert gate["source_executed_lifetime_difference_observed"] is True
    assert gate["mapping_conditioned_capacity_difference_observed"] is False
    assert gate["strong_mapping_conditioned_lifetime_claim_qualified"] is False
    assert gate["novelty_over_prior_art_established"] is False
