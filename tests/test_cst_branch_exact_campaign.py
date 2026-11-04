"""Reproducible source-pinned exact scratch campaign acceptance."""
import csv
import json
import shutil
import pytest

from fpgai_research.experiments.cst_branch_exact_campaign import CASES,run


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_prespecified_exhaustive_source_optimum_and_actual_traces(tmp_path):
    folder=tmp_path/"branch_exact"
    result=run(output=folder)
    assert result["cases"]==2*len(CASES)
    assert result["matched_pairs"]==len(CASES)
    assert result["optimal_e0_proofs"]==len(CASES)
    assert result["executed_host_numerical_passes"]==2*len(CASES)
    assert result["executed_source_trace_passes"]==2*len(CASES)
    assert result["budget_unknown_control"]=="UNKNOWN_BUDGET"
    assert result["native_vitis_calls"]==0
    with (folder/"cases.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==2*len(CASES)
    assert all(row["source_trace_pass"]=="True" for row in rows)
    assert json.loads((folder/"summary.json").read_text())==result
    with pytest.raises(FileExistsError):
        run(output=folder)
