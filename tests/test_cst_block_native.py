"""Original-Vitis row-block sweep preparation and no-fabricated-QoR report."""
import csv
import json

import pytest

from fpgai_research.experiments.cst_block_native_prepare import prepare
from fpgai_research.experiments.cst_block_native_report import report
from fpgai_research.experiments.cst_native_campaign import (
    audit, execute, verify_prepared,
)


def test_block_sweep_prepares_three_matched_source_pinned_native_cases(tmp_path):
    target = tmp_path/"native"
    original = prepare(output=target, max_cases=3)
    assert original["study_family"] == "row_block_schedule_v1"
    assert [r["row_block"] for r in original["cases"]] == [1,2,3]
    assert len({r["cpp_sha256"] for r in original["cases"]}) == 3
    assert len({r["functional_sha256"] for r in original["cases"]}) == 1
    assert original["original_native_tool_executed_by_preparation"] is False
    _, checked = verify_prepared(target)
    assert len(checked) == 3
    assert execute(target, execute_native=False, max_calls=3)["original_vitis_calls"] == 0
    raw = audit(target)
    assert len(raw["cases"]) == 3
    assert all(not r["hls_csim_attested_with_bounded_latency"] for r in raw["cases"])
    table = tmp_path/"scorecard"
    summary = report(target, table)
    assert summary["paired_comparisons"] == 2
    assert summary["pairs_without_required_native_measurements"] == 2
    assert summary["tool_invocations_by_report"] == 0
    rows = list(csv.DictReader((table/"pairs.csv").open()))
    assert [int(r["row_block_variant"]) for r in rows] == [2,3]
    assert all(r["original_native_matched_pair_qualified"] == "False"
               for r in rows)
    assert all(r["hls_cycle_ratio_reference_over_variant"] == ""
               for r in rows)
    assert json.loads((table/"summary.json").read_text()) == summary
    with pytest.raises(FileExistsError, match="(?i:create-only)"):
        prepare(output=target, max_cases=3)
    with pytest.raises(FileExistsError, match="Create-only"):
        report(target, table)


def test_block_sweep_incorrect_case_count_and_constraints_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="3, 6, 9 or 12"):
        prepare(output=tmp_path/"invalid", max_cases=4)
    with pytest.raises(ValueError, match="clock"):
        prepare(output=tmp_path/"invalid2", max_cases=3, clock_ns=6,
                clock_uncertainty_ns=8.64)
