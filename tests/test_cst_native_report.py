"""Paper scorecard preserves missing native evidence and source pairing."""
import csv
import json

import pytest

from fpgai_research.experiments.cst_native_campaign import prepare
from fpgai_research.experiments.cst_native_report import build_scorecard


def test_scorecard_does_not_fabricate_unrun_native_cases(tmp_path):
    study = tmp_path / "study"
    prepare(output=study, max_cases=3)
    output = tmp_path / "report"
    result = build_scorecard(study, output)
    assert result["matched_triplet_groups"] == 1
    assert result["paired_variant_rows"] == 2
    assert result["paired_rows_with_source_attested_numerical_and_cycles"] == 0
    assert result["not_measured_rows"] == 2
    assert result["physical_speedup_claim_allowed"] is False
    assert json.loads((output / "report.json").read_text()) == result
    with (output / "paired_hls.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(x["source_attested_bounded_latency_pair"] == "False" for x in rows)
    assert all(x["hls_report_cycle_ratio_reference_over_variant"] == "" for x in rows)
    with pytest.raises(FileExistsError, match="Create-only"):
        build_scorecard(study, output)
