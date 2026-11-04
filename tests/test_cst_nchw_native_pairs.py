"""Source-only NCHW matched native packages: genuine hashes, no Vitis calls."""
import json
import hashlib
import pytest

from fpgai_research.experiments.cst_nchw_native_pairs import (
    CASES,MAPPINGS,VARIANTS,prepare,audit_prepared,
)


def test_native_source_pairs_and_read_only_audit(tmp_path):
    root=tmp_path/"matched"
    report=prepare(output=root)
    expected=len(CASES)*len(MAPPINGS)*len(VARIANTS)
    assert report["case_count"]==expected
    assert report["matched_source_pairs"]==expected//2
    assert report["generated_vitis_package_count"]==expected
    assert report["original_vitis_calls"]==0
    assert report["native_qualified_cases"]==0
    assert report["csim_evidence_status"]=="NOT_RUN"
    assert report["csynth_evidence_status"]=="NOT_RUN"
    assert audit_prepared(root)["status"]=="SOURCE_ONLY_PACKAGES_VERIFIED"
    rows=report["cases"]
    for i in range(0,len(rows),2):
        lhs,rhs=rows[i:i+2]
        assert lhs["model_sha256"]==rhs["model_sha256"]
        assert lhs["architecture_sha256"]==rhs["architecture_sha256"]
        assert lhs["testbench_sha256"]==rhs["testbench_sha256"]
        assert lhs["tcl_sha256"]==rhs["tcl_sha256"]
        assert lhs["source_cpp_sha256"]!=rhs["source_cpp_sha256"]
        assert lhs["real_scratch_elements"]>rhs["real_scratch_elements"]
        assert (root/lhs["case"]/"run_hls.tcl").read_text()==(
            root/rhs["case"]/"run_hls.tcl").read_text()
        assert "csim_design" in (root/lhs["case"]/"run_hls.tcl").read_text()
        assert "csynth_design" in (root/lhs["case"]/"run_hls.tcl").read_text()
    assert json.loads((root/"native_source_manifest.json").read_text())==report
    with pytest.raises(FileExistsError):
        prepare(output=root)


def test_native_pair_audit_fails_closed_if_cpp_changed(tmp_path):
    root=tmp_path/"native"
    prepare(output=root)
    original=root/"case_000"/"model.cpp"
    original.write_text(original.read_text()+"\n// dishonest source edit\n")
    with pytest.raises(ValueError,match="Tampered"):
        audit_prepared(root)

def test_native_case_manifest_tampering_rejected(tmp_path):
    root=tmp_path/"native"
    prepare(output=root)
    file=root/"case_001"/"manifest.json"
    stored=json.loads(file.read_text())
    stored["clock_ns"]=99.0
    file.write_text(json.dumps(stored))
    with pytest.raises(ValueError,match="Tampered|Canonical"):
        audit_prepared(root)


def test_native_packages_refuse_invalid_timing_or_missing_source(tmp_path):
    for args in ({"clock_ns":0.0},{"clock_ns":8.0,"uncertainty_ns":9.0},
                 {"part":";hack"},{"tolerance":-1.0}):
        with pytest.raises(ValueError):
            prepare(output=tmp_path/("invalid_"+str(len(list(tmp_path.iterdir())))),**args)
    assert list(tmp_path.iterdir())==[]
    with pytest.raises(ValueError,match="root"):
        audit_prepared(tmp_path/"not_created")
