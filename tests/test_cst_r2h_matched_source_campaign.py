"""R2h complete 36 matched pairs/72 GCC source and read-only audit."""
from __future__ import annotations

import csv
from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments.cst_r2h_matched_source_campaign import run
from fpgai_research.experiments.cst_r2h_matched_source_audit import (
    audit_original,main,
)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp("r2h_matched_36_pairs")/"original"
    summary=run(output=root)
    assert summary["prespecified_original_matched_pairs"]==36
    assert summary["original_GCC_compilations"]==72
    assert summary["original_GCC_executions"]==72
    return root


def test_all_36_original_GCC_pairs_numerical_addresses_and_readonly_audit(cohort,tmp_path):
    summary=json.loads((cohort/"summary.json").read_text())
    assert summary["prespecified_graphs"]==9
    assert summary["prespecified_Phi_policies_per_graph"]==4
    assert summary["original_all_matched_source_Phi_numeric_address_passes"]==36
    assert summary["original_unique_Phi_interventions"]==27
    assert (summary["original_Phi_reduced_exact_unit_slots"]+
            summary["original_Phi_tied_exact_unit_slots"]+
            summary["original_Phi_regressed_exact_unit_slots"])==27
    assert (summary["source_scratch_reduced_matched_pairs"]+
            summary["source_scratch_tied_matched_pairs"]+
            summary["source_scratch_regressed_matched_pairs"])==36
    assert summary["source_alternative_temporary_arrays_accounted"] is True
    assert summary["physical_BRAM_or_clock_savings_established"] is False
    assert summary["native_Vitis_calls"]==0
    with (cohort/"cases.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==36
    assert len({(r["fanout"],r["shape"],r["policy"]) for r in rows})==36
    for row in rows:
        folder=cohort/row["case"]
        assert row["original_numeric_and_event_passes"]=="True"
        assert row["actual_HLS_or_BRAM_savings_measured"]=="False"
        for layout,label in (("disjoint","baseline"),
                             ("per_element_optimal","optimized")):
            cpp=folder/(layout+"_source.cpp")
            trace=folder/(layout+"_original_stdout.txt")
            assert sha256(cpp.read_bytes()).hexdigest()==row[label+"_source_sha256"]
            assert sha256(trace.read_bytes()).hexdigest()==row[
                label+"_original_stdout_sha256"]
            assert "FPGAI_SLOT" in trace.read_text()
        assert int(row["baseline_scratch_unit_slots"])>=int(
            row["optimized_scratch_unit_slots"])
    checked=audit_original(cohort)
    assert checked["original_matched_pairs_readonly_verified"]==36
    assert checked["original_source_programs_readonly_verified"]==72
    assert checked["all_actual_runtime_source_addresses_independently_replayed"] is True
    assert checked["all_intermediate_and_final_numerics_independently_recomputed"] is True
    assert checked["source_minimal_scalar_unit_slots_certified_for_fixed_schedule"] is True
    assert checked["original_GCC_compilations_or_executions_during_audit"]==0
    assert checked["original_Vitis_Vivado_calls_during_audit"]==0
    assert checked["physical_BRAM_clock_or_RAM_ports_qualified"] is False
    assert len(checked["cases"])==36
    report=tmp_path/"new_audit"
    main(["--study",str(cohort),"--output",str(report)])
    assert json.loads((report/"r2h_audit.json").read_text())==checked
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(cohort),"--output",str(report)])
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=cohort)


def _copy(cohort,tmp_path):
    target=tmp_path/"copy_for_tamper"
    shutil.copytree(cohort,target)
    return target


def test_auditor_rejects_forged_original_runtime_memory_address(cohort,tmp_path):
    root=_copy(cohort,tmp_path)
    path=root/"case_000"/"per_element_optimal_original_stdout.txt"
    text=path.read_text()
    line=next(v for v in text.splitlines() if v.startswith("FPGAI_SLOT READ"))
    pieces=line.split()
    pieces[-1]=str(int(pieces[-1])+100)
    path.write_text(text.replace(line," ".join(pieces),1))
    with pytest.raises(ValueError,match="source/trace/slot lineage SHA mismatch"):
        audit_original(root)


def test_auditor_rejects_inflated_reported_savings(cohort,tmp_path):
    root=_copy(cohort,tmp_path)
    path=root/"cases.csv"
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    rows[0]["source_scratch_unit_slots_saved"]=str(
        int(rows[0]["source_scratch_unit_slots_saved"])+5)
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="matched-pair accounting"):
        audit_original(root)


def test_auditor_refuses_missing_original_case_even_if_other_cases_pass(cohort,tmp_path):
    root=_copy(cohort,tmp_path)
    (root/"case_021"/"FAILED.json").write_text('{"status":"FAILED_ORIGINAL_R2h"}\n')
    with pytest.raises(ValueError,match="failed, reordered or missing"):
        audit_original(root)
