"""Prespecified generalized NCHW graph source evidence accounting."""
import csv,json,shutil,pytest
from fpgai_research.experiments.cst_nchw_graph_campaign import (
    CASES,POLICIES,LAYOUTS,run,
)

@pytest.mark.skipif(shutil.which("g++") is None,reason="Original GCC required")
def test_full_general_nchw_source_and_logical_layout_denominator(tmp_path):
    root=tmp_path/"evidence"
    report=run(output=root)
    total=len(CASES)*len(POLICIES)*len(LAYOUTS)
    assert report["cases"]==total
    assert report["logical_e0_passes"]==total
    assert report["independent_numpy_e0_passes"]==total
    assert report["original_gcc_numerical_passes"]==total
    assert report["actual_original_gcc_conv_source_load_passes"]==total
    assert report["unique_original_gcc_executions"]==total//2
    assert report["source_matched_logical_layout_pairs"]==total//2
    assert report["real_source_alias_emission_cases"]==0
    assert report["original_vitis_calls"]==0
    with (root/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==total
    assert {x["graph"] for x in rows}=={"branch_concat","pool_fork","broadcast_fork"}
    assert all(x["physical_alias_emitted"]=="False" for x in rows)
    assert json.loads((root/"summary.json").read_text())==report
    with pytest.raises(FileExistsError):
        run(output=root)
