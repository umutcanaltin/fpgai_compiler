"""Prespecified create-only exact allocator negative and positive controls."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_fanout_allocator_campaign import run


def test_fanout_allocator_campaign_keeps_all_denominators(tmp_path):
    path = tmp_path / "fanout_exact"
    report = run(output=path)
    assert report["cases"] == 4
    assert report["optimal_e0"] == 2
    assert report["infeasible_cap"] == 1
    assert report["unknown_budget"] == 1
    assert report["native_vitis_calls"] == 0
    with (path / "cases.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 4
    assert rows[0]["status"] == "OPTIMAL_E0"
    assert rows[2]["status"] == "INFEASIBLE_WITHIN_CAPACITY"
    assert rows[3]["status"] == "UNKNOWN_BUDGET"
    assert json.loads((path / "summary.json").read_text()) == report
    with pytest.raises(FileExistsError):
        run(output=path)
