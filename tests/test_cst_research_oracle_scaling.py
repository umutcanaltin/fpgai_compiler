"""Differential and scaling campaigns preserve full science denominators."""
import csv
import json
import pytest

from fpgai_research.experiments.cst_oracle_campaign import run as oracle_run
from fpgai_research.experiments.cst_scaling_campaign import run as scaling_run


def test_independent_oracle_campaign_records_every_permutation(tmp_path):
    target = tmp_path/"oracle"
    result = oracle_run(output=target, domains=2)
    assert result["complete_permutations"] == 1440
    assert result["reference_legal"] > 0
    assert result["reference_illegal"] > 0
    assert result["false_accepts"] == 0
    assert result["false_rejects"] == 0
    assert result["all_legalities_agree"]
    assert len(result["domains"]) == 2
    assert all(x["tested"] == 720 for x in result["domains"])
    assert json.loads((target/"summary.json").read_text()) == result
    with (target/"cases.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 1440
    with pytest.raises(FileExistsError, match="Create-only"):
        oracle_run(output=target, domains=2)


def test_scaling_study_tracks_raw_timing_samples_and_bounds(tmp_path):
    target = tmp_path/"scale"
    summary = scaling_run(output=target, max_graphs=2, repetitions=2)
    assert summary["distinct_graphs"] == 2
    assert summary["schedule_samples"] == 8
    assert summary["all_schedule_samples_accepted"]
    assert summary["max_checked_element_events"] > 0
    assert summary["raw_timing_samples_preserved"]
    assert summary["vitis_calls"] == 0
    with (target/"cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert all(int(x["schedule_check_ns"]) >= 0 for x in rows)
    assert json.loads((target/"summary.json").read_text()) == summary
    with pytest.raises(FileExistsError, match="Create-only"):
        scaling_run(output=target, max_graphs=2, repetitions=2)


@pytest.mark.parametrize("domains", [0, 6, True])
def test_oracle_study_rejects_invalid_domain_counts(tmp_path, domains):
    with pytest.raises(ValueError, match="domains"):
        oracle_run(output=tmp_path/"bad", domains=domains)


@pytest.mark.parametrize("graphs,repetitions", [(0,1),(16,1),(1,0),(1,11)])
def test_scaling_study_rejects_invalid_budgets(tmp_path, graphs, repetitions):
    with pytest.raises(ValueError, match="max_graphs|repetitions"):
        scaling_run(output=tmp_path/"bad", max_graphs=graphs,
                    repetitions=repetitions)
