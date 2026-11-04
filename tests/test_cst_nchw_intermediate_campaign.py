"""Original intermediate-snapshot campaign keeps complete source denominator."""
import csv,json,shutil
import pytest
from fpgai_research.experiments.cst_nchw_intermediate_campaign import run,CASES,POLICIES


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GCC required")
def test_full_source_level_intermediate_numerical_evidence(tmp_path):
    folder=tmp_path/"all_stages"
    report=run(output=folder)
    expected=len(CASES)*len(POLICIES)
    assert report["cases"]==expected
    assert report["unique_original_gcc_source_executions"]==expected
    assert report["actual_intermediate_snapshot_passes"]==expected
    assert report["total_original_operator_snapshots"]>=expected*5
    assert report["total_checked_intermediate_elements"]>expected*10
    assert report["original_Vitis_calls"]==0
    with (folder/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==expected
    assert all(r["all_intermediates_match_numpy"]=="True" for r in rows)
    assert all(r["actual_global_READ_COMMIT_order_attested"]=="False" for r in rows)
    assert len(json.loads((folder/"operator_receipts.json").read_text()))==expected
    assert json.loads((folder/"summary.json").read_text())==report
    with pytest.raises(FileExistsError):
        run(output=folder)
