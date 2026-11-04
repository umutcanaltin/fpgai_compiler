"""Deterministic control/fault campaign is actual science, not decorative tests."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_schedule_ablation import run


def test_schedule_interventions_are_paired_and_report_complete_denominators(tmp_path):
    target = tmp_path/"schedule"
    outcome = run(output=target, max_graphs=2, host_cases=2, row_trace_cases=2)
    assert outcome["prespecified_graphs"] == 2
    assert outcome["positive_controls"] == 4
    assert outcome["positive_control_accepts"] == 4
    assert outcome["negative_interventions"] == 18
    assert outcome["negative_interventions_correctly_rejected"] == 18
    assert outcome["prespecified_variant_count"] == 22
    assert outcome["host_executed_positives"] == 2
    assert outcome["host_numerical_passes"] == 2
    assert outcome["host_executed_temporal_traces"] == 2
    assert outcome["source_operator_refinement_passed_for_all_interventions"]
    assert outcome["vitis_calls"] == 0
    assert outcome["per_intervention"]["insufficient_scratch_slots"] == {
        "attempts": 2, "recognized": 2,
    }
    rows = list(csv.DictReader((target/"cases.csv").open()))
    assert len(rows) == 22
    assert json.loads((target/"summary.json").read_text()) == outcome
    assert all(r["compositional_schedule_status"] != "VERIFIED"
               for r in rows if r["intervention"] != "valid")
    with pytest.raises(FileExistsError, match="Create-only"):
        run(output=target, max_graphs=2, host_cases=0, row_trace_cases=0)


@pytest.mark.parametrize("n,host,trace", [
    (0, 0, 0), (19, 0, 0), (2, -1, 0), (2, 1, 2), (2, 101, 0)
])
def test_invalid_schedule_campaign_configuration_fails_closed(tmp_path, n, host, trace):
    with pytest.raises(ValueError, match="graphs|budget|Host"):
        run(output=tmp_path/"invalid", max_graphs=n, host_cases=host,
            row_trace_cases=trace)
