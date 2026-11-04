"""R2g complete original matched 36-source-GCC proof and read-only audit."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments.cst_r2g_fanout_source_campaign import run
from fpgai_research.experiments.cst_r2g_fanout_source_audit import (
    audit_original, main,
)


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    output=tmp_path_factory.mktemp("new_R2g_fanout_e1")/"original_36"
    summary=run(output=output)
    assert summary["prespecified_original_cases"]==36
    assert summary["original_GCC_compilations"]==36
    assert summary["original_GCC_executions"]==36
    return output


def test_all_36_GCC_original_cases_and_readonly_independent_audit(original,tmp_path):
    summary=json.loads((original/"summary.json").read_text())
    assert summary["prespecified_fanouts"]==[2,3,4]
    assert summary["prespecified_original_cases"]==36
    assert summary["independent_source_event_Phi_and_alias_passes"]==36
    assert summary["all_intermediate_and_final_NumPy_numeric_passes"]==36
    assert summary["runtime_dynamic_subscript_cases"]==36
    assert summary["source_disjoint_scratch_layout_cases"]==36
    assert summary["actual_Phi_conditioned_memory_capacity_savings_executed"] is False
    assert summary["hardware_cycle_port_or_BRAM_qualified"] is False
    with (original/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==36
    assert len({(r["fanout"],r["shape"],r["policy"]) for r in rows})==36
    for row in rows:
        folder=original/row["case"]
        assert sha256((folder/"source.cpp").read_bytes()).hexdigest()==(
            row["source_cpp_sha256"])
        assert sha256((folder/"original_stdout.txt").read_bytes()).hexdigest()==(
            row["original_stdout_sha256"])
        assert int(row["reader_H_reads"])>0
        assert int(row["producer_contributions"])>0
        assert int(row["intermediate_numeric_values"])>0
        assert row["all_source_oracles_passed"]=="True"
        assert row["actual_per_element_optimized_storage_executed"]=="False"
    checked=audit_original(original)
    assert checked["original_cases_readonly_verified"]==36
    assert checked["independent_dynamic_Phi_and_producer_reader_Add_checks"]==36
    assert checked["all_intermediate_numerics_and_final_NumPy_verified"]==36
    assert checked["original_GCC_compilations_or_executions_during_audit"]==0
    assert checked["native_Vitis_or_Vivado_calls_during_audit"]==0
    assert checked["source_value_layout"]=="DISJOINT_ONLY_NOT_UNIT_SLOT_OPTIMIZED"
    assert len(checked["cases"])==36
    destination=tmp_path/"new_readonly_audit"
    main(["--study",str(original),"--output",str(destination)])
    assert json.loads((destination/"r2g_audit.json").read_text())==checked
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(original),"--output",str(destination)])
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=original)


def _copy(original,tmp_path):
    destination=tmp_path/"source_mutation_copy"
    shutil.copytree(original,destination)
    return destination


def test_original_GCC_trace_is_hash_attested_and_tamper_rejected(original,tmp_path):
    cohort=_copy(original,tmp_path)
    path=cohort/"case_000"/"original_stdout.txt"
    raw=path.read_text()
    assert "FPGAI_R2G READ h" in raw
    path.write_text(raw.replace("FPGAI_R2G READ h",
                                "FPGAI_R2G READ stale",1))
    with pytest.raises(ValueError,match="SHA lineage"):
        audit_original(cohort)


def test_original_36_denominator_cannot_be_inflated(original,tmp_path):
    cohort=_copy(original,tmp_path)
    path=cohort/"summary.json"
    data=json.loads(path.read_text())
    data["all_intermediate_and_final_NumPy_numeric_passes"]=37
    path.write_text(json.dumps(data)+"\n")
    with pytest.raises(ValueError,match="aggregate cohort contradicts cases"):
        audit_original(cohort)


def test_original_case_failure_is_never_silently_dropped(original,tmp_path):
    cohort=_copy(original,tmp_path)
    (cohort/"case_015"/"FAILED.json").write_text(
        '{"status":"FAILED_ORIGINAL_R2g"}\n')
    with pytest.raises(ValueError,match="failed/missing/reordered"):
        audit_original(cohort)
