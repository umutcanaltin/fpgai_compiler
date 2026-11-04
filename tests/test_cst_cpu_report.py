"""CST research paper CPU table evidence must be source matched and explicit."""
from __future__ import annotations

import pytest

from fpgai_research.experiments.cst_ir_campaign import run as run_array
from fpgai_research.experiments.cst_row_campaign import run as run_row
from fpgai_research.experiments.cst_chain_campaign import run as run_chain
from fpgai_research.experiments.cst_cpu_report import report


def test_paper_table_is_from_actual_source_pinned_cpu_runs(tmp_path):
    array = tmp_path/"arrays"
    row = tmp_path/"rows"
    chain = tmp_path/"chains"
    run_array(output=array, max_cases=2, host_check_pairs=0)
    run_row(output=row, max_pairs=1, host_pairs=0)
    run_chain(output=chain, max_cases=1, host_cases=0, full_trace_cases=0)
    target = tmp_path/"paper"
    output = report(array_study=array, row_study=row, chain_study=chain, output=target)
    assert output["original_host_checked_total"] == 0
    assert output["no_vitis_source_evidence_promoted"]
    assert not output["prior_art_novelty_established"]
    doc = (target/"cpu_results.md").read_text()
    assert "Array layout" in doc
    assert "Row lifetime" in doc
    assert "Multi-operator chain" in doc
    assert "Evidence E0/E1 only" in doc
    with pytest.raises(FileExistsError, match="Create-only"):
        report(array_study=array, row_study=row, chain_study=chain, output=target)


def test_report_rejects_tampered_original_case_rows(tmp_path):
    array = tmp_path/"arrays"
    row = tmp_path/"rows"
    chain = tmp_path/"chains"
    run_array(output=array, max_cases=2, host_check_pairs=0)
    run_row(output=row, max_pairs=1, host_pairs=0)
    run_chain(output=chain, max_cases=1, host_cases=0, full_trace_cases=0)
    (array/"cases.csv").write_text("bad,row\n")
    with pytest.raises(ValueError, match="Study summary row counts|array study has no source-pinned rows"):
        report(array_study=array, row_study=row, chain_study=chain,
               output=tmp_path/"paper")
