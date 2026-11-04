"""Prespecified source-pinned real all-access GCC campaign denominator."""
import csv,json,shutil
import pytest
from fpgai_research.experiments.cst_nchw_access_campaign import (
    run,CASES,MAPPINGS,STORAGE,
)

@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_all_real_RW_source_cases_keep_full_denominator(tmp_path):
    folder=tmp_path/"access"
    report=run(output=folder)
    expected=len(CASES)*len(MAPPINGS)*len(STORAGE)
    assert report["cases"]==expected
    assert report["genuine_original_GCC_executions"]==expected
    assert report["source_read_write_verified"]==expected
    assert report["real_shared_source_pairs"]==expected//2
    assert report["minimum_source_scratch_element_saving"]>0
    assert report["observed_real_reads"]>report["observed_real_writes"]
    assert report["original_Vitis_calls"]==0
    assert report["hardware_qualified_cases"]==0
    with (folder/"cases.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==expected
    assert all(r["source_access_order_verified"]=="True" for r in rows)
    assert json.loads((folder/"summary.json").read_text())==report
    with pytest.raises(FileExistsError):
        run(output=folder)
