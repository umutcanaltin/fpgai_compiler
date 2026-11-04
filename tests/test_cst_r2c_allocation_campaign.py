"""Prospective R2c exact scratch E1 matched original source campaign."""
from __future__ import annotations

import csv
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments.cst_r2c_allocation_campaign import run
from fpgai_research.experiments.cst_r2c_allocation_audit import audit_original,main


@pytest.fixture
def original(tmp_path):
    output=tmp_path/"original_r2c_source_pinned"
    summary=run(output=output)
    assert summary["prespecified_original_matched_pairs"]==12
    assert summary["original_GCC_executions"]==24
    return output


def test_full_twelve_matched_GCC_original_source_pairs_and_independent_reaudit(
        original,tmp_path):
    original_summary=json.loads((original/"summary.json").read_text())
    assert original_summary["original_numeric_and_typed_trace_pair_passes"]==12
    assert original_summary["original_slot_witness_pair_passes"]==12
    assert original_summary["original_Vitis_calls"]==0
    assert original_summary["original_Vivado_calls"]==0
    assert original_summary["min_unit_slots_independently_witnessed_by_peak_live_counts"]==12
    assert original_summary["source_declared_mutable_array_elements_counted"] is True
    with (original/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==12
    for row in rows:
        source_dir=original/row["case"]
        assert (source_dir/"baseline_source.cpp").is_file()
        assert (source_dir/"allocated_source.cpp").is_file()
        assert sha256((source_dir/"allocated_source.cpp").read_bytes()).hexdigest()==(
            row["allocated_source_sha256"])
        assert sha256((source_dir/"allocated_original_stdout.txt").read_bytes()).hexdigest()==(
            row["allocated_trace_sha256"])
        assert int(row["declared_mutable_array_elements_saved"])==(
            int(row["source_scratch_slots_saved"]))
        assert int(row["allocated_per_element_scratch_slots"])<=int(
            row["baseline_whole_tensor_scratch_slots"])
    audited=audit_original(original)
    assert audited["original_source_matched_pairs_readonly_qualified"]==12
    assert audited["original_GCC_case_sources_readonly_qualified"]==24
    assert audited["original_GCC_executions_this_audit"]==0
    assert audited["original_Vitis_or_Vivado_calls_this_audit"]==0
    assert audited["all_executed_source_address_witnesses_verified"] is True
    assert audited["all_original_outputs_independently_recomputed"] is True
    assert audited["all_original_exact_source_minimality_bounds_verified"] is True
    assert audited["all_five_declared_mutable_source_arrays_included"] is True
    assert audited["hardware_BRAM_or_clock_performance_qualified"] is False
    assert audited["original_unique_Phi_interventions"]==9
    assert len(audited["original_pairs"])==12
    result_dir=tmp_path/"independent_result"
    main(["--study",str(original),"--output",str(result_dir)])
    assert json.loads((result_dir/"r2c_audit.json").read_text())==audited
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(original),"--output",str(result_dir)])
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=original)


def test_independent_auditor_rejects_forged_original_GCC_dynamic_scratch_slot(original):
    path=original/"case_000"/"allocated_original_stdout.txt"
    trace=path.read_text()
    line=next(row for row in trace.splitlines() if row.startswith("FPGAI_SLOT READ"))
    fields=line.split()
    fields[-1]=str(int(fields[-1])+1)
    path.write_text(trace.replace(line," ".join(fields),1))
    with pytest.raises(ValueError,match="source/trace SHA-256"):
        audit_original(original)


def test_independent_auditor_refuses_inflated_source_storage_savings(original):
    csv_path=original/"cases.csv"
    with csv_path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    rows[0]["source_scratch_slots_saved"]=str(
        int(rows[0]["source_scratch_slots_saved"])+3)
    with csv_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="source scratch and declared array accounting"):
        audit_original(original)


def test_original_missing_or_failed_case_is_not_silently_dropped(original):
    (original/"case_003"/"FAILED.json").write_text('{"status":"FAILED"}\n')
    with pytest.raises(ValueError,match="Missing, reordered or failed"):
        audit_original(original)
