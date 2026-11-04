"""Read-only R2b causal re-audit must not double-count layout variants."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_r2b_source_campaign import run
from fpgai_research.experiments.cst_r2b_causal_audit import (
    evaluate_original, main,
)


@pytest.fixture
def original(tmp_path):
    output=tmp_path/"untouched_GCC_case_population"
    report=run(output=output)
    assert report["prespecified_cases"]==24
    return output


def test_independent_original_causal_reanalysis_collapses_pseudoreplication(
        original, tmp_path):
    report=evaluate_original(original)
    assert report["prespecified_original_GCC_source_cases"]==24
    assert report["original_source_cases_read_only_reverified"]==24
    assert report["prespecified_source_storage_layout_pairs"]==12
    assert report["layout_replicated_phi_comparisons"]==18
    assert report["UNIQUE_graph_Phi_interventions"]==9
    assert report["global_event_index_changed_unique_interventions"]==9
    # 12/18 layout-specific effects represent 6/9 distinct interventions.
    assert report["H_only_final_read_rank_or_consumer_changed_unique_interventions"]==6
    assert report["H_only_final_read_rank_or_consumer_unchanged_unique_interventions"]==3
    # 6/18 final consumer effects represent 3/9 distinct interventions.
    assert report["final_consumer_identity_changed_unique_interventions"]==3
    assert report["original_reported_layout_scoped_read_changes"]==12
    assert report["both_layout_controls_numerically_and_event_equivalent"] is True
    assert report["actual_mapping_conditional_source_scratch_capacity_established"] is False
    assert report["source_level_Phi_aware_consumption_order_effect_established"] is True
    assert report["hardware_cycle_or_FPGA_QoR_effect_established"] is False
    assert report["original_GCC_compilations_or_executions_this_audit"]==0
    assert report["original_vendor_calls_this_audit"]==0
    assert len(report["unique_interventions"])==9
    assert len(report["original_layout_controls"])==12
    assert {row["policy"] for row in report["unique_interventions"]}=={
        "reader0_pes2","reader1_pes2","both_pes2"}
    assert all(row["two_layout_controls_agree"]
               for row in report["unique_interventions"])

    # Omit output => read-only. Otherwise a newly named create-only report.
    destination=tmp_path/"independent_results"
    main(["--study",str(original)])
    assert not destination.exists()
    main(["--study",str(original),"--output",str(destination)])
    saved=json.loads((destination/"r2b_causal_audit.json").read_text())
    assert saved==report
    with pytest.raises(FileExistsError,match="create-only"):
        main(["--study",str(original),"--output",str(destination)])


def test_causal_audit_rejects_summary_mismatch_even_with_valid_case_receipts(original):
    path=original/"summary.json"
    summary=json.loads(path.read_text())
    summary["source_final_reader_identity_changed_contrasts"]=18
    path.write_text(json.dumps(summary)+"\n")
    with pytest.raises(ValueError,match="contradicts independent raw causal tallies"):
        evaluate_original(original)


def test_causal_audit_rejects_modified_input_even_if_all_event_hashes_still_pass(original):
    case=original/"case_000"/"original_case.json"
    record=json.loads(case.read_text())
    record["input_float32_values"][0]+=1
    case.write_text(json.dumps(record)+"\n")
    with pytest.raises(ValueError,match="numerical input changed"):
        evaluate_original(original)


def test_causal_audit_rejects_duplicate_graph_phi_storage_case(original):
    path=original/"cases.csv"
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    rows[1]["shape"]=rows[0]["shape"]
    rows[1]["policy"]=rows[0]["policy"]
    rows[1]["layout"]=rows[0]["layout"]
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The first independent source/trace auditor must reject an exact
    # duplicate before the secondary causal checker begins its analysis.
    with pytest.raises(ValueError,match="Duplicate original R2b source case"):
        evaluate_original(original)


def test_causal_audit_rejects_unrecognized_mapping_key_after_primary_source_audit(original):
    path=original/"cases.csv"
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    # Keep a unique case ID and tuple so the immutable source auditor can
    # verify its original stdout; the higher-level fixed protocol still
    # must reject an unregistered mapping intervention.
    rows[1]["policy"]="unsupported_phi_policy"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError,match="Unexpected or duplicate original graph/mapping/layout"):
        evaluate_original(original)
