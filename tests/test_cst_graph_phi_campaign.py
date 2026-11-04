"""Prespecified bounded generalized DAG source experimental denominator."""
import csv
import json
import shutil
import pytest

from fpgai_research.experiments.cst_graph_phi_campaign import run,CASES,POLICIES


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC original host needed")
def test_nested_graph_source_study_exhaustively_retains_paired_cases(tmp_path):
    folder=tmp_path/"graph_phi"
    summary=run(output=folder)
    n=2*len(CASES)*len(POLICIES)
    assert summary["cases"]==n
    assert summary["original_numeric_passes"]==n
    assert summary["original_source_trace_passes"]==n
    assert summary["paired_source_layouts"]==n//2
    assert summary["actual_original_vitis_calls"]==0
    with (folder/"cases.csv").open() as inp:
        rows=list(csv.DictReader(inp))
    assert len(rows)==n
    assert {int(x["operator_count"]) for x in rows}>={6,7,8}
    assert len({x["functional_sha256"] for x in rows})==len(CASES)
    assert json.loads((folder/"summary.json").read_text())==summary
    with pytest.raises(FileExistsError):
        run(output=folder)
