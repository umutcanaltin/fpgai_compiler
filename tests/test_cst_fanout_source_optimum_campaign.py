"""Exact source-order E0 allocator and E1 host controls preserve all pairs."""
import csv
import json
import shutil

import pytest

from fpgai_research.experiments.cst_fanout_source_optimum_campaign import run


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_source_order_optimum_campaign_has_full_case_denominators(tmp_path):
    folder=tmp_path/"fanout_source_optimum"
    result=run(output=folder)
    assert result["cases"]==6
    assert result["paired_graphs"]==3
    assert result["successful_host_numeric"]==6
    assert result["successful_source_traces"]==6
    assert result["source_optimality_certificates"]==3
    assert result["source_scratch_saving_pairs"]==3
    assert result["native_vitis_calls"]==0
    with (folder/"cases.csv").open() as fp:
        cases=list(csv.DictReader(fp))
    assert len(cases)==6
    for a,b in zip(cases[::2],cases[1::2]):
        assert a["functional_sha256"]==b["functional_sha256"]
        assert int(a["source_scratch_elements"])>int(b["source_scratch_elements"])
        assert a["source_e0_optimum_slots"]==b["source_e0_optimum_slots"]
    assert json.loads((folder/"summary.json").read_text())==result
    with pytest.raises(FileExistsError):
        run(output=folder)
