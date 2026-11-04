"""R3l: complete revision-pinned original publication evidence aggregation.

The test constructs *receipt fixtures* and monkeypatches source auditors.
It does not impersonate these fixtures as new original experiments.
Actual R3l CLI must receive the author's unchanged on-disk original files.
"""
from __future__ import annotations

import json
from hashlib import sha256

import pytest

from fpgai_research.experiments import cst_r3l_c1c2_publication_release as sut
from fpgai_research.experiments import cst_r3c_cnn_publication_tables as r3c


@pytest.fixture
def synthetic_original_receipts(tmp_path,monkeypatch):
    build=tmp_path/"build"
    build.mkdir()
    r3c_path=build/sut.DEFAULT_FOLDER["R3c"]
    r3c_path.mkdir()
    pair=lambda status:{
        "both_E3":status,
        "cycles_control":624 if status else None,
        "cycles_independent":911 if status else None,
        "cycles_delta":287 if status else None,
        "estimated_latency_delta_ns":2870 if status else None,
        "estimated_latency_change_percent":46.0 if status else None,
        "bram_18k_delta":0 if status else None,
        "dsp_delta":48 if status else None,
        "lut_delta":9146 if status else None,
        "ff_delta":1796 if status else None,
    }
    data={
        "schema":r3c.SCHEMA,
        "source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3c"],
        "original_native_attempts":6,
        "original_CSim_E2_qualified":6,
        "original_CSynth_E3_qualified":4,
        "original_CSynth_E3_rejected":2,
        "matched_pairs_prespecified":3,
        "matched_pairs_both_E3_qualified":2,
        "qualified_pairs_independent_phi_more_cycles":2,
        "qualified_pairs_estimated_BRAM_ties":2,
        "cases":[{"case":f"case_{i:03d}"} for i in range(6)],
        "pairs":[pair(True),pair(False),pair(True)],
        "physical_E4":None,
        "learning_improvement":None,
        "scientific_novelty_established":False,
        "original_vendor_calls_during_analysis":0,
    }
    (r3c_path/"summary.json").write_text(json.dumps(data))
    (r3c_path/"paper_results.md").write_text("source-only synthetic fixture")
    (r3c_path/"case_population.csv").write_text(
        "case\n"+"\n".join(f"case_{i:03d}" for i in range(6))+"\n")
    (r3c_path/"matched_pair_population.csv").write_text(
        "both_E3\nTrue\nFalse\nTrue\n")
    originals=("summary.json","paper_results.md","case_population.csv",
               "matched_pair_population.csv")
    digests={name:sha256((r3c_path/name).read_bytes()).hexdigest()
             for name in originals}
    (r3c_path/"derived_export_sha256.json").write_text(json.dumps(digests))
    fake={
        "R3d":{
            "source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3d"],
            "finite_candidate_denominator":6,"e0_certified":3,
            "rejected":2,"unknown_e0":1,
            "saved_receipt_sha256":"a"*64,
            "original_compiler_or_vendor_calls_during_audit":0,
        },
        "R3e":{
            "source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3e"],
            "original_case_count":5,
            "rejected_mutants":25,
            "independent_minimum_slot_oracle_agreements":5,
            "read_only_receipt_sha256":"b"*64,
            "new_compiler_gcc_vitis_vivado_calls":0,
        },
        "R3f":{
            "original_source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3f"],
            "candidate_population":48,"certified":21,"rejected":24,
            "unknown_e0":3,"read_only_receipt_sha256":"c"*64,
            "new_compiler_or_vendor_calls":0,
        },
        "R3g":{
            "source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3g"],
            "tested_event_trace_variants":133,
            "independent_oracle_disagreements":0,
            "source_receipt_sha256":"d"*64,
            "new_compiler_gcc_vitis_vivado_calls":0,
        },
        "R3h":{
            "source_revision":sut.ORIGINAL_SOURCE_REVISIONS["R3h"],
            "candidate_population":48,"catalog_compiler_calls":48,
            "compiled_once_selection_additional_compiler_calls":0,
            "source_receipt_sha256":"e"*64,
            "new_compiler_or_vendor_calls_during_audit":0,
        },
    }
    for key in sut.REQUIRED:
        if key!="R3c":
            (build/sut.DEFAULT_FOLDER[key]).mkdir()
    called=[]
    def fake_reader(key):
        def read(path):
            called.append((key,path))
            return fake[key]
        return read
    monkeypatch.setattr(sut,"READERS",
                        {k:fake_reader(k) for k in fake})
    return build,r3c_path,data,fake,called


def test_complete_required_original_source_cohorts_without_experiment_calls(
    synthetic_original_receipts,tmp_path
):
    build,_,_,fake,called=synthetic_original_receipts
    result=sut.gather(build)
    assert result["schema"]==sut.SCHEMA
    assert result["submission_ready"] is False
    assert result["decision"].startswith("MAJOR_REVISION")
    assert set(result["original_cohorts"])==set(sut.REQUIRED)
    assert all(item["status"]=="VERIFIED_ORIGINAL_RECEIPT"
               for item in result["original_cohorts"].values())
    assert len(called)==len(fake)==5
    assert result["native_r3c_e2"]==6
    assert result["native_r3c_e3"]==4
    assert result["native_r3c_matched_e3"]==2
    assert result["c1_checked_source_trace_variants_r3g"]==133
    assert result["c1_r3g_oracle_disagreements"]==0
    assert result["c2_r3f_e0_unknown"]==3
    assert result["c2_r3h_cached_selection_extra_transitions"]==0
    out=tmp_path/"publication_release"
    sut.export_create_only(result,output=out)
    assert (out/"evidence_manifest.json").is_file()
    assert "MAJOR REVISION" in (out/"paper_evidence_table.md").read_text()
    assert "R3h" in (out/"original_cohort_provenance.csv").read_text()
    digests=json.loads((out/"derived_sha256.json").read_text())
    assert all(digests[key]==sha256((out/key).read_bytes()).hexdigest()
               for key in digests)
    with pytest.raises(FileExistsError,match="create-only"):
        sut.export_create_only(result,output=out)


def test_optional_unexecuted_is_not_silently_promoted(synthetic_original_receipts):
    build,*_=synthetic_original_receipts
    result=sut.gather(build,include_optional=True)
    assert result["original_cohorts"]["R3j"]["audit"] is None
    assert result["original_cohorts"]["R3k"]["audit"] is None
    assert result["original_cohorts"]["R3j"]["status"]==(
        "UNEXECUTED_OPTIONAL_SOURCE_EXPERIMENT")
    assert result["submission_ready"] is False


def test_missing_original_cohort_fails_closed(synthetic_original_receipts):
    build,*_=synthetic_original_receipts
    (build/sut.DEFAULT_FOLDER["R3g"]).rmdir()
    with pytest.raises(ValueError,match="absent"):
        sut.gather(build)


def test_wrong_original_source_revision_fails_closed(
    synthetic_original_receipts
):
    build,_,_,fake,_=synthetic_original_receipts
    fake["R3h"]["source_revision"]="f"*40
    with pytest.raises(ValueError,match="does not match frozen cohort"):
        sut.gather(build)


def test_modified_native_R3c_derived_file_digest_rejected(
    synthetic_original_receipts
):
    build,r3c_path,*_=synthetic_original_receipts
    (r3c_path/"paper_results.md").write_text("post-hoc amended data")
    with pytest.raises(ValueError,match="digest mismatch"):
        sut.gather(build)


def test_rejected_native_hls_pair_must_keep_null_resource_and_cycles(
    synthetic_original_receipts
):
    build,r3c_path,data,*_=synthetic_original_receipts
    changed=dict(data)
    changed["pairs"]=[dict(p) for p in data["pairs"]]
    changed["pairs"][1]["cycles_delta"]=7
    summary=r3c_path/"summary.json"
    summary.write_text(json.dumps(changed))
    manifest=r3c_path/"derived_export_sha256.json"
    all_digests=json.loads(manifest.read_text())
    all_digests["summary.json"]=sha256(summary.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(all_digests))
    with pytest.raises(ValueError,match="imputed QoR"):
        sut.gather(build)


def test_r3d_source_action_population_cannot_be_promoted(
    synthetic_original_receipts
):
    build,_,_,fake,_=synthetic_original_receipts
    fake["R3d"]["e0_certified"]=4
    with pytest.raises(ValueError,match="deviates from original"):
        sut.gather(build)


def test_source_report_cannot_claim_submission_ready(
    synthetic_original_receipts,tmp_path
):
    build,*_=synthetic_original_receipts
    result=sut.gather(build)
    with pytest.raises(ValueError,match="could not be independently"):
        sut.export_create_only(
            {**result,"submission_ready":True},
            output=tmp_path/"false_submission")
