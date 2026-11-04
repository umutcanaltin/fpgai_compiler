"""Paper-first row-lifetime CPU experiment regression tests, no vendor calls."""
import csv
import json

import pytest

from fpgai_research.experiments.cst_row_campaign import run


def test_row_lifetime_campaign_is_create_only_matched_and_source_only(tmp_path):
    output = tmp_path / "row-cst"
    result = run(output, max_pairs=4, host_pairs=0)
    assert result["paired_cases"] == 4
    assert result["host_checked_pairs"] == 0
    assert result["original_vitis_calls"] == 0
    assert result["all_generated_sources_distinct"]
    assert result["max_source_scratch_reduction_ratio"] in (2, 3, 5)
    assert json.loads((output / "summary.json").read_text())["paired_cases"] == 4
    with (output / "cases.csv").open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 4
    assert all(int(r["original_vitis_calls"]) == 0 for r in records)
    assert all(r["baseline_cpp_sha256"] != r["row_interleaved_cpp_sha256"]
               for r in records)
    with pytest.raises(FileExistsError, match="Create-only"):
        run(output, max_pairs=4, host_pairs=0)


def test_row_campaign_hard_input_limits(tmp_path):
    with pytest.raises(ValueError, match="max_pairs"):
        run(tmp_path / "bad", max_pairs=0, host_pairs=0)
    with pytest.raises(ValueError, match="host_pairs"):
        run(tmp_path / "bad2", max_pairs=2, host_pairs=3)
