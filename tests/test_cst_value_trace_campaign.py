"""Scientific E1 per-element readiness and last-use witness campaign tests."""
from __future__ import annotations
import csv
import json
import pytest

from fpgai_research.experiments.cst_value_trace_campaign import run


def test_actual_source_mapped_value_campaign_preserves_all_cases(tmp_path):
    output=tmp_path/"source_values"
    result=run(output=output,max_cases=2)
    assert result["cases_attempted"]==2
    assert result["source_value_trace_passes"]==2
    assert result["paired_phi_control"]["same_functional_graph"]
    assert result["paired_phi_control"]["same_row_exchange_schedule"]
    assert result["paired_phi_control"]["different_actual_mapping_and_source_traces"]
    assert result["total_mapped_reduction_iterations"]>0
    assert result["total_verified_consumer_releases"]>0
    assert result["original_vitis_calls"]==0
    with (output/"cases.csv").open() as data:
        rows=list(csv.DictReader(data))
    assert len(rows)==2
    assert all(x["source_verified"]=="True" for x in rows)
    assert json.loads((output/"summary.json").read_text())==result
    with pytest.raises(FileExistsError,match="Create-only"):
        run(output=output,max_cases=1)


@pytest.mark.parametrize("count",[0,7,-1,True])
def test_value_trace_campaign_rejects_invalid_sample_count(tmp_path,count):
    with pytest.raises(ValueError,match="max_cases"):
        run(output=tmp_path/"bad",max_cases=count)
