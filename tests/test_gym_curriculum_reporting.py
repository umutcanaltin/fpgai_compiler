"""Fixture-only safety tests for stage reporting and XML clock diagnostics."""
from pathlib import Path
import json

import pytest

from rl_environment.clock_mismatch_probe import investigate
from rl_environment.gym_curriculum_report import export


def test_clock_mismatch_probe_reads_original_xml_values_without_vendor_calls(
        tmp_path):
    study = tmp_path / "old"
    case = study / "seed_111/train/train_00000/episode_000000/case_0000"
    report = case / "fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.xml"
    report.parent.mkdir(parents=True)
    report.write_text(
        "<Profile><UserAssignments><TargetClockPeriod>10</TargetClockPeriod>"
        "<ClockUncertainty>1</ClockUncertainty></UserAssignments>"
        "<PerformanceEstimates><SummaryOfTimingAnalysis>"
        "<EstimatedClockPeriod>5</EstimatedClockPeriod>"
        "</SummaryOfTimingAnalysis></PerformanceEstimates></Profile>")
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 9.125}))
    (study / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "source_revision": "a" * 40,
    }))
    (study / "native_steps.jsonl").write_text(json.dumps({
        "private_episode_path": str(case.parent),
        "clock_ns": 9.125,
        "native_call_index": 0,
    }) + "\n")
    result = investigate(study)
    assert result["native_vendor_calls"] == 0
    assert result["source_files_modified"] == 0
    assert result["counts"]["mismatching_target_clock"] == 1
    assert result["example_mismatches"][0]["reported_xml_target_clock_ns"] == 10.
    assert result["example_mismatches"][0]["requested_manifest_clock_ns"] == 9.125
    assert result["example_mismatches"][0][
        "difference_target_minus_requested_ns"] == pytest.approx(.875)
    assert result["top_clock_pairs_requested_reported"][0]["count"] == 1


def test_curriculum_scorecard_refuses_incomplete_or_unpinned_studies(tmp_path):
    from rl_environment.gym_curriculum_report import scorecard
    with pytest.raises(FileNotFoundError):
        scorecard(tmp_path / "not_a_native_experiment")
    with pytest.raises(ValueError, match="at least one"):
        export([], tmp_path / "never_written")
    assert not (tmp_path / "never_written").exists()
