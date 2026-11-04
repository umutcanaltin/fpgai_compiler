"""R3h C2: compiled-once source catalog campaign and immutable audit tests."""
from __future__ import annotations

import json
import pytest

from fpgai_research.experiments import cst_r3h_c2_compiled_once_selection as sut
from fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage import COHORTS


@pytest.fixture(scope="module")
def complete_catalog_report():
    return sut.run()


@pytest.mark.parametrize("name,width", COHORTS)
def test_cached_action_selection_retains_original_compiler_receipt(name,width):
    result=sut.evaluate(name,width)
    assert result["cohort"]==name
    assert result["parameter"]==width
    assert result["catalog_compiler_calls"]==8
    assert result["legacy_mask_compiler_calls"]==8
    assert result["legacy_second_compile_if_selected"]==1
    assert result["catalog_additional_compile_if_selected"]==0
    assert (result["certified_action_count"]+
            result["unknown_e0_count"]+
            result["rejected_count"])==8
    assert result["certified_action_count"]>0
    assert len(result["selected_source_cpp_sha256"])==64
    assert len(result["selected_next_state_sha256"])==64
    assert result["native_vitis_calls"]==0
    assert result["trained_agents"]==0
    assert result["catalog_wall_ns"]>=0
    assert result["cached_selection_wall_ns"]>=0


def test_prespecified_C2_finite_catalog_calls_and_negative_cost_limit(complete_catalog_report):
    study=complete_catalog_report
    assert study["schema"]==sut.SCHEMA
    assert study["original_graph_population"]==6
    assert study["original_candidate_population"]==48
    assert study["original_compiled_source_selection_population"]==6
    assert study["catalog_source_compiler_calls"]==48
    assert study["independent_legacy_mask_compiler_calls"]==48
    assert study["catalog_additional_selection_compiler_calls"]==0
    assert study["legacy_additional_selection_compiler_calls"]==6
    assert study["trained_rl_policies"]==0
    assert study["new_vitis_calls"]==0
    assert "single blind unmasked proposal" in " ".join(study["limits"]).lower()


def test_immutable_report_create_only_and_saved_evidence_audit(tmp_path,complete_catalog_report):
    result=complete_catalog_report
    out=tmp_path/"original"
    sut.create_only(result,output=out,source_revision="a"*40)
    report=sut.audit(out)
    assert report["candidate_population"]==48
    assert report["catalog_compiler_calls"]==48
    assert report["compiled_once_selection_additional_compiler_calls"]==0
    assert report["new_compiler_or_vendor_calls_during_audit"]==0
    assert len(report["source_receipt_sha256"])==64
    with pytest.raises(FileExistsError,match="create-only"):
        sut.create_only(result,output=out,source_revision="a"*40)


@pytest.mark.parametrize("mutate",[
    lambda r: r.update(catalog_source_compiler_calls=47),
    lambda r: r["rows"][0].update(catalog_additional_compile_if_selected=1),
    lambda r: r["rows"][1].update(selected_source_cpp_sha256="wrong"),
    lambda r: r["rows"][2].update(chosen_action_index=88),
    lambda r: r.update(trained_rl_policies=1),
    lambda r: r.update(source_revision="not-a-full-sha"),
])
def test_saved_catalog_selection_audit_refuses_unjustified_claim(tmp_path,mutate,complete_catalog_report):
    study=complete_catalog_report
    output=tmp_path/"evidence"
    sut.create_only(study,output=output,source_revision="b"*40)
    file=output/"r3h_compiled_once_selection.json"
    data=json.loads(file.read_text())
    mutate(data)
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="R3h:"):
        sut.audit(output)
