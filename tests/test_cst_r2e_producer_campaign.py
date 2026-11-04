"""R2e prospective original 12-case GCC producer/reader/numeric cohort tests."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments.cst_r2e_producer_campaign import run
from fpgai_research.experiments.cst_r2e_producer_audit import (
    audit_original,main,
)


@pytest.fixture(scope="module")
def full_cohort(tmp_path_factory):
    study=tmp_path_factory.mktemp("r2e_complete_finite_study")/"original_12"
    result=run(output=study)
    assert result["prespecified_original_cases"]==12
    assert result["original_GCC_compilations"]==12
    assert result["original_GCC_executions"]==12
    return study


def test_complete_R2e_original_GCC_producer_matmul_and_readonly_audit(full_cohort,tmp_path):
    original=json.loads((full_cohort/"summary.json").read_text())
    assert original["prespecified_graphs"]==3
    assert original["prespecified_reader_Phi_policies"]==4
    assert original["original_producer_H_commit_complete_cases"]==12
    assert original["original_numeric_and_consumer_source_passes"]==12
    assert original["original_intermediate_H_numeric_passes"]==12
    assert original["original_Vitis_calls"]==0
    with (full_cohort/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==12
    for row in rows:
        folder=full_cohort/row["case"]
        assert sha256((folder/"source.cpp").read_bytes()).hexdigest()==(
            row["source_cpp_sha256"])
        assert sha256((folder/"original_runtime_trace.txt").read_bytes()).hexdigest()==(
            row["original_trace_sha256"])
        assert float(row["original_H_numeric_max_abs_error"])<1e-4
        assert row["runtime_H_values_sha256"]
        assert int(row["actual_producer_contributions"])>0
        assert int(row["actual_complete_H_producer_commits"])>0
        assert row["source_producer_complete"]=="True"
    audit=audit_original(full_cohort)
    assert audit["original_source_cases_readonly_verified"]==12
    assert audit["complete_H_producer_and_typed_COMMIT_cases"]==12
    assert audit["independent_Phi_reader_issue_passes"]==12
    assert audit["independent_NumPy_numeric_passes"]==12
    assert audit["independent_intermediate_H_numeric_passes"]==12
    assert audit["FPGA_hardware_producer_completeness_proven"] is False
    assert audit["original_GCC_compilations_or_executions_during_audit"]==0
    assert len(audit["cases"])==12
    output=tmp_path/"audit_result"
    main(["--study",str(full_cohort),"--output",str(output)])
    assert json.loads((output/"r2e_audit.json").read_text())==audit
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(full_cohort),"--output",str(output)])
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=full_cohort)


def _clone(full_cohort,tmp_path):
    target=tmp_path/"tampered_copy"
    shutil.copytree(full_cohort,target)
    return target


def test_R2e_audit_rejects_saved_source_stdout_hash_tampering(full_cohort,tmp_path):
    study=_clone(full_cohort,tmp_path)
    file=study/"case_000"/"original_runtime_trace.txt"
    raw=file.read_text()
    assert "FPGAI_PROD CONTRIBUTE" in raw
    file.write_text(raw.replace("FPGAI_PROD CONTRIBUTE",
                                "FPGAI_PROD BADCONTRIBUTE",1))
    with pytest.raises(ValueError,match="SHA or semantic identity"):
        audit_original(study)


def test_R2e_audit_rejects_aggregate_inflated_numeric_pass_count(full_cohort,tmp_path):
    study=_clone(full_cohort,tmp_path)
    file=study/"summary.json"
    report=json.loads(file.read_text())
    report["original_numeric_and_consumer_source_passes"]=13
    file.write_text(json.dumps(report)+"\n")
    with pytest.raises(ValueError,match="aggregate summary contradicts"):
        audit_original(study)


def test_R2e_audit_rejects_an_original_failed_source_case(full_cohort,tmp_path):
    study=_clone(full_cohort,tmp_path)
    (study/"case_004"/"FAILED.json").write_text(
        '{"status":"FAILED_ORIGINAL_R2E_PRODUCER"}\n')
    with pytest.raises(ValueError,match="failed/reordered/missing"):
        audit_original(study)
