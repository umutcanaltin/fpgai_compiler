"""New source-pinned rank-two broadcast/Concat model-IR study denominator."""
import csv
import json
import shutil
import pytest

from fpgai_research.experiments.cst_graph_access_campaign import run,CASES,POLICIES


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_access_map_campaign_complete_and_create_only(tmp_path):
    folder=tmp_path/"access_map"
    summary=run(output=folder)
    expected=2*len(CASES)*len(POLICIES)
    assert summary["cases"]==expected
    assert summary["source_numerical_passes"]==expected
    assert summary["source_trace_passes"]==expected
    assert summary["safe_layout_pairs"]==expected//2
    assert summary["original_vitis_calls"]==0
    with (folder/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==expected
    assert {r["graph"] for r in rows}=={"broadcast","concat"}
    assert json.loads((folder/"summary.json").read_text())==summary
    with pytest.raises(FileExistsError):
        run(output=folder)
