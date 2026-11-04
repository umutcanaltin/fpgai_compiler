"""Real host GCC fan-out campaign is source-pinned, paired and create-only."""
import csv
import json
import shutil

import pytest

from fpgai_research.experiments.cst_fanout_source_campaign import run


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_fanout_e1_campaign_does_not_infer_native_hardware(tmp_path):
    root=tmp_path/"fanout_e1"
    result=run(output=root)
    assert result["cases"]==6
    assert result["distinct_graphs"]==3
    assert result["paired_source_layouts"]==3
    assert result["actual_gcc_numerical_passes"]==6
    assert result["actual_gcc_trace_passes"]==6
    assert result["source_level_scratch_reduction_pairs"]==3
    assert result["actual_vitis_cases"]==0
    with (root/"cases.csv").open() as fp:
        records=list(csv.DictReader(fp))
    assert len(records)==6
    for left,right in zip(records[::2],records[1::2]):
        assert left["functional_sha256"]==right["functional_sha256"]
        assert left["cpp_sha256"]!=right["cpp_sha256"]
        assert int(left["scratch_source_elements"])>int(right["scratch_source_elements"])
    assert json.loads((root/"summary.json").read_text())==result
    with pytest.raises(FileExistsError):
        run(output=root)
