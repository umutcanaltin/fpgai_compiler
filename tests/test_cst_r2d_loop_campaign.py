"""R2d original fixed 12 runtime-index C++ cases + independent evidence audit."""
from __future__ import annotations

import csv
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments.cst_r2d_loop_campaign import run
from fpgai_research.experiments.cst_r2d_loop_audit import (
    audit_original, main,
)


@pytest.fixture
def original(tmp_path):
    study=tmp_path/"original_r2d_runtime_loop"
    result=run(output=study)
    assert result["prespecified_cases"]==12
    assert result["original_GCC_compilations"]==12
    assert result["original_GCC_executions"]==12
    return study


def test_original_12_loop_sources_pass_independent_Phi_and_numeric_audit(
        original,tmp_path):
    original_summary=json.loads((original/"summary.json").read_text())
    assert original_summary["prespecified_graphs"]==3
    assert original_summary["prespecified_Phi_reader_policies"]==4
    assert original_summary["runtime_index_loop_cases"]==12
    assert original_summary["independent_runtime_Phi_issue_passes"]==12
    assert original_summary["source_reduction_alias_passes"]==12
    assert original_summary["original_numeric_passes"]==12
    assert original_summary["hardware_cycle_schedule_proven"] is False
    assert original_summary["original_Vitis_calls"]==0
    with (original/"cases.csv").open(newline="") as stream:
        cases=list(csv.DictReader(stream))
    assert len(cases)==12
    for row in cases:
        folder=original/row["case"]
        assert (folder/"source.cpp").is_file()
        assert (folder/"original_gcc_trace.txt").is_file()
        assert sha256((folder/"source.cpp").read_bytes()).hexdigest()==(
            row["cpp_sha256"])
        assert sha256((folder/"original_gcc_trace.txt").read_bytes()).hexdigest()==(
            row["original_stdout_sha256"])
    checked=audit_original(original)
    assert checked["original_cases_reverified"]==12
    assert checked["original_runtime_loop_Phi_issue_passes"]==12
    assert checked["original_reduction_alias_passes"]==12
    assert checked["original_independent_numeric_passes"]==12
    assert checked["original_GCC_compilations_or_executions_this_audit"]==0
    assert checked["original_native_vendor_calls_this_audit"]==0
    assert checked["independent_spec_not_generator_event_replay"] is True
    assert len(checked["cases"])==12
    report_dir=tmp_path/"r2d_independent_audit"
    main(["--study",str(original),"--output",str(report_dir)])
    assert json.loads((report_dir/"r2d_audit.json").read_text())==checked
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(original),"--output",str(report_dir)])
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=original)


def test_tampered_runtime_index_witness_fails_read_only_audit(original):
    file=original/"case_002"/"original_gcc_trace.txt"
    data=file.read_text()
    assert "FPGAI_R2B READ h " in data
    file.write_text(data.replace("FPGAI_R2B READ h ", "FPGAI_R2B READ stale ",1))
    with pytest.raises(ValueError,match="SHA chain invalid"):
        audit_original(original)


def test_inflated_source_numeric_pass_count_is_rejected(original):
    file=original/"summary.json"
    data=json.loads(file.read_text())
    data["original_numeric_passes"]=13
    file.write_text(json.dumps(data)+"\n")
    with pytest.raises(ValueError,match="summary is inconsistent"):
        audit_original(original)


def test_failed_original_case_cannot_be_silently_omitted(original):
    (original/"case_005"/"FAILED.json").write_text(
        '{"status":"FAILED_ORIGINAL_R2D_E1"}\n')
    with pytest.raises(ValueError,match="Missing, reordered or failed"):
        audit_original(original)
