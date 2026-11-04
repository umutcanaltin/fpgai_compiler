"""R2f original 36-case E0 fanout source/capacity study regression."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS,SHAPES,POLICY_NAMES,policy_widths,run,
)


def test_exact_prespecified_fanout_Phi_population_and_artifact_controls(tmp_path):
    result=run()
    assert result["prespecified_fanout_degrees"]==[2,3,4]
    assert result["prespecified_cases"]==3*3*4==36
    assert result["all_producer_reader_Add_and_final_E0_certificates_verified"]==36
    assert result["independent_Phi_issue_certificates_verified"]==36
    assert result["exact_fixed_order_unit_slot_certificates_verified"]==36
    assert result["unique_graph_Phi_interventions"]==27
    assert (result["Phi_interventions_reduced_logical_unit_slots"]+
            result["Phi_interventions_tied_logical_unit_slots"]+
            result["Phi_interventions_regressed_logical_unit_slots"])==27
    assert 0<=result["changed_unit_slot_capacity_interventions"]<=27
    assert 0<=result["changed_H_last_reader_rank_interventions"]<=27
    assert result["original_GCC_executions"]==0
    assert result["original_Vitis_calls"]==0
    assert result["source_code_generation_or_executable_validation_established"] is False
    assert result["hardware_scratch_or_bram_savings_established"] is False

    output=tmp_path/"R2f_E0_original"
    report=run(output=output)
    assert report==result
    assert json.loads((output/"summary.json").read_text())==report
    with (output/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==36
    assert len({(r["fanout"],r["shape"],r["policy"]) for r in rows})==36
    for row in rows:
        assert row["source_readiness_verified"]=="True"
        assert row["independent_Phi_issue_verified"]=="True"
        assert row["logical_scalar_alias_replay_verified"]=="True"
        assert row["source_C++_or_original_GCC_executed"]=="False"
        assert row["native_vendor_calls"]=="0"
        fanout=int(row["fanout"])
        m,k,n,q=json.loads(row["shape"])
        assert int(row["producer_contributions"])==m*k*n
        assert int(row["H_reads"])==fanout*m*n*q
        assert int(row["logical_intermediate_values"])==m*n+(2*fanout-1)*m*q
        assert int(row["source_peak_live_unit_values"])==int(
            row["exact_fixed_order_minimum_unit_slots"])
        assert int(row["whole_tensor_disjoint_unit_slots"])>=int(
            row["exact_fixed_order_minimum_unit_slots"])
        assert row["functional_sha256"]
        assert row["architecture_sha256"]
        assert row["source_event_sha256"]
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=output)


def test_r2f_reader_Phi_policies_do_not_silently_rewrite_widths():
    assert policy_widths(2,"scalar")==(1,1)
    assert policy_widths(3,"reader0_pes2")==(2,1,1)
    assert policy_widths(4,"last_reader_pes2")==(1,1,1,2)
    assert policy_widths(4,"all_pes2")==(2,2,2,2)
    with pytest.raises(ValueError,match="prespecified"):
        policy_widths(5,"scalar")
    with pytest.raises(ValueError,match="prespecified"):
        policy_widths(2,"unsupported")
