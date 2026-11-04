"""CPU-only completion protocol; synthetic statuses are NEVER E4 evidence."""
from __future__ import annotations

import copy
import json

import pytest

from fpgai_research.experiments import cst_research_completion as completion


def fixture_cases(*,finished=0,broken=None):
    cases=[]
    for i in range(12):
        passed=i<finished
        cases.append({
            "case":f"case_{i:03d}",
            "group":i//2,
            "graph":("branch_concat","pool_fork","broadcast_fork")[i//4],
            "mapping":"independent_phi" if (i//2)%2 else "scalar",
            "source_storage":"real_disjoint" if i%2==0 else "real_shared",
            "requested_clock_ns":32.0,
            "original_hls_qualified":True,
            "status":("QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4" if passed
                      else "PREPARED_NOT_RUN"),
            "routed_qualified":passed,
            "routed_warning_observation":False,
            "routed_observation_valid":passed,
        })
    if broken is not None:
        cases[broken]["status"]="UNKNOWN_OR_FAILED_VIVADO_DRC"
        cases[broken]["routed_qualified"]=False
        cases[broken]["routed_observation_valid"]=False
    return {"prespecified_cases":12,"cases":cases,
            "vivado_invocation_receipts":finished}


def test_completion_plan_is_source_matched_zero_vendor_calls(monkeypatch):
    current=fixture_cases(finished=2)
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:current)
    def reject_native(*args,**kwargs):
        raise AssertionError("Read-only plan may never invoke vendor tools")
    monkeypatch.setattr(completion,"run_routed_study",reject_native)
    p=completion.completion_plan("original_study")
    assert p["prepared_cases"]==10
    assert p["strict_e4_cases"]==2
    assert p["next_prepared_case"]=="case_002"
    assert p["first_blocking_case"] is None
    assert p["allowed_to_execute_one_original_case"] is True
    assert p["publication_ready"] is False
    assert p["original_vendor_calls_this_command"]==0


def test_original_rejected_case_blocks_automatic_progress(monkeypatch):
    current=fixture_cases(finished=2,broken=2)
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:current)
    monkeypatch.setattr(completion,"run_routed_study",lambda **_:(
        pytest.fail("A rejected original case must not rerun")))
    p=completion.completion_plan("original_study")
    assert p["first_blocking_case"]["case"]=="case_002"
    with pytest.raises(ValueError,match="never rerun"):
        completion.continue_physical(
            study="original_study",confirm_isolated_native=True)


def test_resume_only_prepared_cases_and_stop_after_source_verified_warning(
    monkeypatch
):
    current=fixture_cases(finished=2)
    calls=[]
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:copy.deepcopy(current))
    def native(study,*,execute_native,confirm_isolated_native,start_case,
               max_cases,vivado_executable,timeout_seconds):
        assert execute_native and confirm_isolated_native and max_cases==1
        assert timeout_seconds==14400
        calls.append(start_case)
        row=current["cases"][start_case]
        assert row["status"]=="PREPARED_NOT_RUN"
        row["status"]="OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
        row["routed_warning_observation"]=True
        row["routed_observation_valid"]=True
        current["vivado_invocation_receipts"]+=1
        return {"new_vivado_calls":1,"cases":[{"case":row["case"]}]}
    monkeypatch.setattr(completion,"run_routed_study",native)
    with pytest.raises(ValueError,match="explicit confirmation"):
        completion.continue_physical(study="original_study")
    result=completion.continue_physical(
        study="original_study",max_new_calls=2,confirm_isolated_native=True)
    assert calls==[2,3]
    assert result["original_vendor_calls_this_command"]==2
    assert result["strict_e4_cases"]==2
    assert result["warning_observation_cases"]==2
    assert result["execution_status"]=="BOUNDED_ORIGINAL_CALL_BUDGET_REACHED"
    assert result["publication_ready"] is False


def test_fail_closed_after_one_rejected_case(monkeypatch):
    current=fixture_cases()
    calls=[]
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:copy.deepcopy(current))
    def native(study,*,start_case,**kwargs):
        calls.append(start_case)
        row=current["cases"][start_case]
        row["status"]="UNKNOWN_OR_FAILED_VIVADO_DRC"
        current["vivado_invocation_receipts"]+=1
        return {"new_vivado_calls":1,"cases":[{"case":row["case"]}]}
    monkeypatch.setattr(completion,"run_routed_study",native)
    result=completion.continue_physical(
        study="original_study",max_new_calls=12,confirm_isolated_native=True)
    assert calls==[0]
    assert result["execution_status"]=="STOPPED_ON_UNQUALIFIED_CASE"
    assert result["prepared_cases"]==11
    with pytest.raises(ValueError,match="never rerun"):
        completion.continue_physical(
            study="original_study",confirm_isolated_native=True)


def test_not_every_hls_qualified_refuses_native_campaign(monkeypatch):
    current=fixture_cases()
    current["cases"][5]["original_hls_qualified"]=False
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:current)
    with pytest.raises(ValueError,match="Incomplete original HLS"):
        completion.continue_physical(
            study="original_study",confirm_isolated_native=True)


def test_original_pair_geometry_and_source_order_must_match(monkeypatch):
    current=fixture_cases()
    current["cases"][1]["mapping"]="different_phi"
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:current)
    with pytest.raises(ValueError,match="source-matched"):
        completion.completion_plan("original_study")


@pytest.mark.parametrize("bad",[0,-1,13,True,1.5])
def test_only_bounded_integer_vivado_budgets(monkeypatch,bad):
    with pytest.raises(ValueError,match="1..12"):
        completion.continue_physical(
            study="irrelevant",max_new_calls=bad,confirm_isolated_native=True)


def test_final_paper_export_retains_primary_and_descriptive_denominators(
    monkeypatch
):
    current=fixture_cases(finished=12)
    monkeypatch.setattr(completion,"audit_routed_study",lambda *_:current)
    def fake_export(*,study,output):
        assert study=="original_study" and output=="fresh_report"
        return {"routed_qualified_pairs":6,"routed_observation_valid_pairs":6}
    monkeypatch.setattr(completion,"export_routed_evidence",fake_export)
    report=completion.export_paper(study="original_study",output="fresh_report")
    assert report["strict_e4_pairs"]==6
    assert report["exploratory_routed_pairs"]==6
    assert report["publication_ready"] is False
    assert report["original_vendor_calls_this_command"]==0
    assert report["novelty_and_prior_art_baselines"]=="NOT_ESTABLISHED"
