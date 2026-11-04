"""Keep every real source scratch comparison in the denominator."""
import csv,json,shutil,pytest
from fpgai_research.experiments.cst_nchw_shared_scratch_campaign import run,CASES,POLICIES

@pytest.mark.skipif(shutil.which("g++") is None,reason="Original GCC required")
def test_all_materialized_source_scratch_pairs(tmp_path):
    folder=tmp_path/"shared"
    result=run(output=folder)
    expected=2*len(CASES)*len(POLICIES)
    assert result["cases"]==expected
    assert result["real_original_gcc_executions"]==expected*2
    assert result["actual_source_scratch_saving_pairs"]==expected//2
    assert result["minimum_actual_C++_scratch_element_saving"]>0
    assert result["all_intermediate_numerical_checks"]==expected
    assert result["all_conv_actual_load_checks"]==expected
    assert result["hardware_qualified_cases"]==0
    assert result["native_vitis_calls"]==0
    with (folder/"cases.csv").open() as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==expected
    assert all(x["real_original_gcc_intermediate_numerical_pass"]=="True"
               for x in rows)
    assert json.loads((folder/"summary.json").read_text())==result
    with pytest.raises(FileExistsError):
        run(output=folder)
