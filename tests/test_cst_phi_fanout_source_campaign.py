"""Source-pinned causal Phi mapping experiment; GCC, no original Vitis."""
import csv
import json
import shutil
import pytest

from fpgai_research.experiments.cst_phi_fanout_source_campaign import run


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_prespecified_source_phi_fanout_runs_and_preserves_raw_denominator(tmp_path):
    root=tmp_path/"mapped_fanout"
    result=run(output=root)
    assert result["cases"]==3
    assert result["actual_host_numeric_pass"]==3
    assert result["actual_mapped_phi_trace_pass"]==3
    assert result["original_vitis_calls"]==0
    with (root/"cases.csv").open() as fp:
        records=list(csv.DictReader(fp))
    assert len(records)==3
    assert all(r["actual_source_trace_pass"]=="True" for r in records)
    assert all(r["numerical_pass"]=="True" for r in records)
    assert all(int(r["phi_iterations"])==2*int(r["m"])*int(r["n"])
               for r in records)
    assert json.loads((root/"summary.json").read_text())==result
    with pytest.raises(FileExistsError):
        run(output=root)
