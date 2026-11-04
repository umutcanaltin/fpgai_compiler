"""R2v source-attested HLS review and complete conditional comparison gates.

No Vitis/Vivado invocations; synthetic test data exercise denominators
but must never be represented as original HLS measurements.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json

import pytest

from fpgai_research.experiments import cst_r2u_hls_mechanism_prepare as prep
from fpgai_research.experiments import cst_r2u_hls_mechanism_execute as native
from fpgai_research.experiments import cst_r2v_hls_factorial_review as sut


def _synthetic_hls_case_audit():
    # These estimated BRAM numbers reflect the user's pasted R2u output
    # for regression of classification ONLY; the synthesized rows are
    # NOT vendor receipts or reproducible scientific evidence.
    bram = (
        0,1,1,1,None,1,1,1,0,1,1,1,None,1,1,1,
        3,3,3,3,3,3,3,3,2,3,3,3,2,3,3,3,
    )
    source = "a"*40
    cases=[]
    for i,arm in enumerate(prep.ARMS * len(prep.GRAPHS)):
        is_e3 = bram[i] is not None
        levels = dict(zip(prep.FACTOR_NAMES, arm))
        cases.append({
            "case":f"case_{i:03d}",
            "graph_index":i//len(prep.ARMS),
            "source_cpp_sha256":f"{i+1:064x}",
            "original_E2_qualified":True,
            "original_E3_qualified":is_e3,
            "actual_native_invocation_receipt":True,
            "qualification_status":(
                "QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL" if is_e3
                else "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
            ),
            "estimated_resources":{
                key:(bram[i] if key=="bram_18k" else 20) if is_e3 else None
                for key in native.RESOURCES
            },
            "hls_worst_cycles":100+i if is_e3 else None,
            **levels,
        })
    contrasts=[]
    for f in prep.FACTOR_NAMES:
        contrasts.extend(native._contrast(cases,f))
    return {
        "schema":native.SCHEMA,
        "original_source_revision": source,
        "planned_source_cases":prep.CASES,
        "observed_source_cases":len(cases),
        "original_vitis_invocation_receipts_present":prep.CASES,
        "native_CSim_E2_qualified_cases":prep.CASES,
        "native_CSynth_E3_qualified_cases":prep.CASES-2,
        "prespecified_matched_factor_contrasts":len(contrasts),
        "original_vendor_calls_during_audit":0,
        "physical_BRAM_measurements":None,
        "physical_Fmax_measurements":None,
        "cases":cases,
        "contrasts":contrasts,
    }


def test_r2v_all_original_case_and_pair_denominators():
    report=sut.review(_synthetic_hls_case_audit(),
                      expected_source_revision="a"*40)
    assert report["original_vitis_invocation_receipts_present"]==32
    assert report["original_E2_CSim_qualified_cases"]==32
    assert report["original_E3_CSynth_qualified_cases"]==30
    assert report["original_E3_unqualified_cases"]==2
    assert [c["case"] for c in report["original_E3_unqualified_case_details"]]==[
        "case_004","case_012"
    ]
    assert report["prespecified_conditional_HLS_factor_pairs"]==64
    assert report["E3_qualified_conditional_factor_pairs"]==57
    assert report["E3_unknown_conditional_factor_pairs"]==7
    assert len(report["by_factor_and_graph"])==8
    assert all(x["planned_conditional_pairs"]==8
               for x in report["by_factor_and_graph"])
    assert sum(x["qualified_bram_comparisons"]
               for x in report["by_factor_and_graph"])==57
    assert report["physical_BRAM"] is None
    assert report["physical_Fmax"] is None
    assert report["new_original_Vitis_calls_during_review"]==0
    assert report["new_original_Vivado_calls_during_review"]==0


def test_r2v_mixed_estimates_never_misclassify_failed_e3_as_bram_tie():
    result=sut.review(_synthetic_hls_case_audit(),
                      expected_source_revision="a"*40)
    ignored=[c for c in result["contrasts"] if not c["e3_both_qualified"]]
    assert len(ignored)==7
    assert all(c["classification"]=="UNKNOWN_NOT_BOTH_E3" for c in ignored)
    assert all(c["bram_18k_b_minus_a"] is None for c in ignored)
    assert all(c["worst_cycles_b_minus_a"] is None for c in ignored)
    assert any(c["classification"]=="REDUCED_HLS_ESTIMATED_BRAM"
               for c in result["contrasts"])
    assert any(c["classification"]=="INCREASED_HLS_ESTIMATED_BRAM"
               for c in result["contrasts"])


def test_tampered_r2u_source_sha_and_qualification_denominators_fail():
    original=_synthetic_hls_case_audit()
    forged=deepcopy(original)
    forged["cases"][0]["source_cpp_sha256"]="invalid"
    with pytest.raises(ValueError,match="provenance"):
        sut.review(forged,expected_source_revision="a"*40)
    forged=deepcopy(original)
    forged["native_CSynth_E3_qualified_cases"]=32
    with pytest.raises(ValueError,match="denominator"):
        sut.review(forged,expected_source_revision="a"*40)


def test_modified_delta_or_factor_pair_rejected():
    base=_synthetic_hls_case_audit()
    forged=deepcopy(base)
    first=next(c for c in forged["contrasts"]
               if c["matched_original_CSynth_qualified"])
    first["estimated_resource_delta_b_minus_a"]["bram_18k"]=222
    with pytest.raises(ValueError,match="resource delta"):
        sut.review(forged,expected_source_revision="a"*40)
    forged=deepcopy(base)
    forged["contrasts"][0]["arm_b"]="case_031"
    with pytest.raises(ValueError,match="paired HLS"):
        sut.review(forged,expected_source_revision="a"*40)


@pytest.fixture(scope="module")
def source_only_r2u_packages(tmp_path_factory):
    folder=tmp_path_factory.mktemp("r2v_no_hls")/"prepared"
    prep.prepare(output=folder)
    return folder


def test_original_source_packages_all_remain_unknown_without_vendor_calls(
        source_only_r2u_packages):
    root=source_only_r2u_packages
    original=native.audit_native(study=root,require_current_source=False)
    assert original["native_CSynth_E3_qualified_cases"]==0
    report=sut.review(
        original,
        expected_source_revision=original["original_source_revision"]
    )
    assert report["original_vitis_invocation_receipts_present"]==0
    assert report["original_E2_CSim_qualified_cases"]==0
    assert report["original_E3_CSynth_qualified_cases"]==0
    assert report["E3_qualified_conditional_factor_pairs"]==0
    assert report["E3_unknown_conditional_factor_pairs"]==64


def test_read_only_study_and_create_only_export(
        source_only_r2u_packages,tmp_path):
    source=source_only_r2u_packages
    before={
        str(p.relative_to(source)):p.read_bytes()
        for p in source.rglob("*") if p.is_file()
    }
    folder=tmp_path/"review"
    report=sut.run(
        study=source,output=folder,
        expected_source_revision=prep._rev()
    )
    assert report["original_E3_CSynth_qualified_cases"]==0
    assert report["original_attested_csynth_xml_reports_present"]==0
    assert len(report["raw_csynth_report_provenance"])==32
    assert (folder/"summary.json").is_file()
    with (folder/"contrasts.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==64
    assert all(row["classification"]=="UNKNOWN_NOT_BOTH_E3" for row in rows)
    assert before=={
        str(p.relative_to(source)):p.read_bytes()
        for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError,match="create-only"):
        sut.run(study=source,output=folder,
                expected_source_revision=prep._rev())


def test_r2v_refuses_wrong_original_revision_without_creating_output(
        source_only_r2u_packages,tmp_path):
    output=tmp_path/"uncreated"
    with pytest.raises(ValueError,match="provenance"):
        sut.run(study=source_only_r2u_packages,
                output=output,expected_source_revision="a"*40)
    assert not output.exists()
