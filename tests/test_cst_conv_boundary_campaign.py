"""Retain complete NCHW Conv E0/E1 numeric scientific denominator."""
import csv,json,shutil,pytest
from fpgai_research.experiments.cst_conv_boundary_campaign import CASES,POLICIES,run

@pytest.mark.skipif(shutil.which("g++") is None,reason="Original GCC required")
def test_all_conv_source_cases_and_source_trace_limit(tmp_path):
    output=tmp_path/"conv"
    receipt=run(output=output)
    count=len(CASES)*len(POLICIES)
    assert receipt["cases"]==count
    assert receipt["independent_conv_e0_pass"]==count
    assert receipt["original_gcc_numeric_pass"]==count
    assert receipt["logical_alias_and_early_reject_pass"]==count
    assert receipt["original_gcc_per_read_trace_pass"]==count
    assert receipt["native_vitis_calls"]==0
    with (output/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==count
    assert all(row["source_read_trace_verified"]=="True" for row in rows)
    assert json.loads((output/"summary.json").read_text())==receipt
    with pytest.raises(FileExistsError):
        run(output=output)
