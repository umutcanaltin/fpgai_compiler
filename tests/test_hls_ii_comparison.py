from __future__ import annotations

import json
from pathlib import Path

from fpgai.analysis.hls_ii_comparison import (
    requested_ii_by_layer,
    write_requested_achieved_ii_summary,
)


def test_requested_ii_by_layer_supports_dict_plan() -> None:
    plan = {
        "layer_plans": [
            {
                "name": "dense0",
                "architecture": {
                    "pipeline": {
                        "ii": 2,
                    }
                },
            },
            {
                "name": "dense1",
                "pipeline_ii": 1,
            },
        ]
    }

    assert requested_ii_by_layer(plan) == {
        "dense0": 2,
        "dense1": 1,
    }


def test_write_requested_achieved_ii_summary(tmp_path: Path) -> None:
    report_path = (
        tmp_path
        / "proj"
        / "solution1"
        / "syn"
        / "report"
        / "deeplearn_csynth.xml"
    )
    report_path.parent.mkdir(parents=True)

    report_path.write_text(
        """
        <profile>
          <PerformanceEstimates>
            <SummaryOfLoopLatency>
              <Loop>
                <Name>dense0_mac_loop</Name>
                <PipelineII>2</PipelineII>
                <AchievedII>2</AchievedII>
              </Loop>
              <Loop>
                <Name>dense1_mac_loop</Name>
                <PipelineII>1</PipelineII>
                <AchievedII>3</AchievedII>
              </Loop>
            </SummaryOfLoopLatency>
          </PerformanceEstimates>
        </profile>
        """,
        encoding="utf-8",
    )

    plan = {
        "layer_plans": [
            {
                "name": "dense0",
                "pipeline_ii": 2,
            },
            {
                "name": "dense1",
                "pipeline_ii": 1,
            },
        ]
    }

    result = write_requested_achieved_ii_summary(
        tmp_path,
        plan,
    )

    assert result is not None
    assert result["path"] == "hls_ii_comparison.json"
    assert result["summary"]["report_count"] == 1
    assert result["summary"]["layer_count"] == 2
    assert result["summary"]["matched_layer_count"] == 2
    assert result["summary"]["failed_layer_count"] == 1

    data = json.loads(
        (tmp_path / "hls_ii_comparison.json").read_text(encoding="utf-8")
    )

    assert data["requested_by_layer"] == {
        "dense0": 2,
        "dense1": 1,
    }

    layers = {
        item["layer_name"]: item
        for item in data["layers"]
    }

    assert layers["dense0"]["ii_met"] is True
    assert layers["dense1"]["ii_met"] is False


def test_compiler_records_ii_comparison_in_manifest_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "write_requested_achieved_ii_summary" in source
    assert "hls_ii_comparison=hls_ii_comparison" in source
    assert 'manifest["hls_ii_comparison"] = hls_ii_comparison' in source
