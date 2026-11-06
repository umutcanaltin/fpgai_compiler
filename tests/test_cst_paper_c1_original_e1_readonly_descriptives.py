"""Original full C1 E1 real-source descriptive postanalysis: synthetic SHA fixtures.

The test constructs fake source/trace receipts in a temporary directory:
these are NOT the author's scientific result. No GCC, Vitis, Vivado,
compiler action replay, GitHub Actions, or original evidence mutation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from fpgai_research.experiments import (
    cst_paper_c1_original_e1_readonly_descriptives as sut,
)

SHA = "a"*64


def _write(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return sha256(path.read_bytes()).hexdigest()


def synthetic_full_e1(tmp_path):
    """A complete but FAKE 271-artifact 54-profile E1 output for tests."""
    study = tmp_path/"synthetic_c1_full"
    study.mkdir()
    cases = []
    files = {}
    for index in range(54):
        seed = sut.SEEDS[index//18]
        readers = sut.READERS[(index%18)//6]
        mapping = sut.MAPPINGS[(index%6)//3]
        shape = [1 + index%3, 2+index%2, 2+(index+1)%3, 2+(index%3)]
        # Original source slots are NOT actual hardware BRAM.
        full_slots = 50 + 10*readers + index%3
        reused_slots = full_slots - (2+readers)
        folder = study/f"case_{index:03d}"
        folder.mkdir()
        arms = {}
        for layout, slots in (("disjoint", full_slots),
                              ("per_element_optimal", reused_slots)):
            source_sha = _write(
                folder/f"{layout}.cpp",
                f"// synthetically constructed {index}, {layout}\n")
            stdout_sha = _write(
                folder/f"{layout}.stdout.txt",
                f"synthetic non-native trace {index}, {layout}\n")
            arms[layout] = {
                "status": "R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED",
                "source_scratch_unit_slots": slots,
                "original_actual_address_witness_count": 5+index,
                "intermediate_numeric_max_absolute_error": 1e-8,
                "final_numeric_max_absolute_error": 2e-8,
                "generated_cpp_sha256": source_sha,
                "actual_stdout_sha256": stdout_sha,
                "runtime_source_event_sha256": "b"*64,
                "functional_sha256": "c"*64,
                "architecture_sha256": "d"*64,
                "input_sha256": "e"*64,
                "original_GCC_compilations": 1,
                "original_GCC_executions": 1,
            }
        case = {
            "seed": seed, "readers": readers, "mapping": mapping,
            "shape": shape,
            "original_c1_e0_event_sha256": "f"*64,
            "new_e1_reuse_source_slot_savings": full_slots-reused_slots,
            "layouts": arms,
            "claims": "bounded actual generated GCC scratch/Numpy source only",
        }
        _write(folder/"case.json",json.dumps(case,sort_keys=True)+"\n")
        cases.append(case)
    summary = {
        "schema": sut.E1_SCHEMA,
        "status": "NEW_E1_EXECUTED_GCC_SOURCE_REFINEMENT_PAIRED_WITH_OLD_E0",
        "experiment_source_revision": sut.ORIGINAL_E1_REVISION,
        "original_E0_source_revision": sut.ORIGINAL_E0_REVISION,
        "original_E0_report_sha256": sut.ORIGINAL_E0_SHA,
        "profiles": 54,
        "smoke_only": False,
        "layouts_per_profile": 2,
        "original_c1_e0_profiles_attested": 54,
        "qualified_GCC_compilations": 108,
        "qualified_GCC_executions": 108,
        "all_original_C1_event_hashes_matched": True,
        "actual_source_slot_savings_count": 54,
        "actual_source_slot_tie_count": 0,
        "native_Vitis_calls": 0, "native_Vivado_calls": 0,
        "physical_compute_latency": None, "board_qualified": False,
        "cases": cases,
    }
    _write(study/"summary.json",json.dumps(summary,indent=2,sort_keys=True)+"\n")
    for path in sorted(study.rglob("*")):
        if path.is_file():
            files[path.relative_to(study).as_posix()] = sha256(
                path.read_bytes()).hexdigest()
    assert len(files)==271
    receipt_sha = _write(
        study/"artifact_sha256.json",
        json.dumps(files,sort_keys=True,indent=2)+"\n")
    return study, receipt_sha


def test_complete_source_only_original_style_analysis(tmp_path):
    root, digest = synthetic_full_e1(tmp_path)
    result = sut.analyze(root,pinned_manifest_sha256=digest)
    assert result["schema"] == sut.SCHEMA
    assert result["profiles"] == 54
    assert result["source_artifacts_attested"] == 271
    assert result["emitted_GCC_executions"] == 108
    assert result["new_compiler_GCC_or_vendor_calls"] == 0
    assert result["physical_BRAM_reduction"] is None
    assert result["physical_inference_latency"] is None
    assert len(result["by_reader_and_mapping"]) == 6
    assert all(x["n_descriptive_profiles"] == 9
               for x in result["by_reader_and_mapping"])
    assert all(x["all_source_slot_contrasts_positive"]
               for x in result["by_reader_and_mapping"])
    assert result["source_profile_aggregates"]["all_source_slot_contrasts_positive"]
    assert result["source_profile_aggregates"]["sum_saved_source_slots_over_distinct_models"] == sum(
        c["saved_slots"] for c in result["per_profile"])
    assert result["source_profile_aggregates"]["max_final_numeric_absolute_error"] == 2e-8


def test_fake_report_without_real_author_manifest_fails_closed(tmp_path):
    root,digest=synthetic_full_e1(tmp_path)
    assert digest != sut.ORIGINAL_MANIFEST_SHA256
    with pytest.raises(ValueError,match="manifest SHA mismatch"):
        sut.analyze(root)


def test_tampered_generated_cpp_detected_before_any_summary(tmp_path):
    root,digest=synthetic_full_e1(tmp_path)
    p=root/"case_020/per_element_optimal.cpp"
    p.write_text(p.read_text()+"// source tampered\n")
    # The author-pinned SHA may be verified against the original manifest,
    # but the actual source hash must still agree with its manifest.
    with pytest.raises(ValueError,match="source or receipts changed"):
        sut.analyze(root,pinned_manifest_sha256=digest)


def test_duplicate_profile_rejected_even_if_manifest_resigned_synthetically(tmp_path):
    root,_=synthetic_full_e1(tmp_path)
    summary=json.loads((root/"summary.json").read_text())
    summary["cases"][1]["shape"] = summary["cases"][0]["shape"]
    summary["cases"][1]["seed"] = summary["cases"][0]["seed"]
    summary["cases"][1]["readers"] = summary["cases"][0]["readers"]
    summary["cases"][1]["mapping"] = summary["cases"][0]["mapping"]
    _write(root/"summary.json",json.dumps(summary,sort_keys=True,indent=2)+"\n")
    c1=root/"case_001/case.json"
    _write(c1,json.dumps(summary["cases"][1],sort_keys=True)+"\n")
    files={}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="artifact_sha256.json":
            files[p.relative_to(root).as_posix()]=sha256(p.read_bytes()).hexdigest()
    new_digest=_write(root/"artifact_sha256.json",
                      json.dumps(files,sort_keys=True,indent=2)+"\n")
    with pytest.raises(ValueError,match="duplicated"):
        sut.analyze(root,pinned_manifest_sha256=new_digest)


@pytest.mark.parametrize("field,value",[
    ("new_e1_reuse_source_slot_savings",0),
    ("new_e1_reuse_source_slot_savings",-1),
])
def test_synthetic_bad_source_slot_receipt_rejected(field,value,tmp_path):
    root,digest=synthetic_full_e1(tmp_path)
    # Direct in-memory case validation is independent of original disk SHA.
    c=json.loads((root/"case_001/case.json").read_text())
    c[field]=value
    with pytest.raises(ValueError,match="source allocation improvement"):
        sut._profile(c)


@pytest.mark.parametrize("field,value",[
    ("original_actual_address_witness_count",0),
    ("source_scratch_unit_slots",-1),
    ("intermediate_numeric_max_absolute_error",float("nan")),
    ("final_numeric_max_absolute_error",float("inf")),
])
def test_bad_e1_numeric_or_witness_fields_rejected(field,value,tmp_path):
    root,_=synthetic_full_e1(tmp_path)
    c=json.loads((root/"case_001/case.json").read_text())
    c["layouts"]["per_element_optimal"][field]=value
    with pytest.raises(ValueError,match="source values/witness"):
        sut._profile(c)


def test_cli_summary_only_hides_rows_without_changing_source_receipts(tmp_path,monkeypatch,capsys):
    root,digest=synthetic_full_e1(tmp_path)
    monkeypatch.setattr(sut,"ORIGINAL_MANIFEST_SHA256",digest)
    sut.main(["--study",str(root),"--summary-only"])
    summary=json.loads(capsys.readouterr().out)
    assert summary["profiles"]==54
    assert "per_profile" not in summary
    assert len(summary["by_reader_and_mapping"])==6
    assert summary["new_compiler_GCC_or_vendor_calls"]==0
    assert sut._sha(root/"artifact_sha256.json")==digest


def test_read_only_original_source_files_unchanged_after_analysis(tmp_path):
    root,digest=synthetic_full_e1(tmp_path)
    before={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}
    sut.analyze(root,pinned_manifest_sha256=digest)
    after={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest()
           for p in root.rglob("*") if p.is_file()}
    assert before==after
