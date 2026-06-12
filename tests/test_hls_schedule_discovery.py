from __future__ import annotations

import json
from pathlib import Path

from fpgai.analysis.hls_schedule_report import (
    discover_hls_schedule_reports,
    summarize_hls_schedule_reports,
    write_hls_schedule_summary,
)


def _write_xml(path: Path, name: str, requested: int, achieved: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
        <profile>
          <PerformanceEstimates>
            <SummaryOfLoopLatency>
              <Loop>
                <Name>{name}</Name>
                <PipelineII>{requested}</PipelineII>
                <AchievedII>{achieved}</AchievedII>
              </Loop>
            </SummaryOfLoopLatency>
          </PerformanceEstimates>
        </profile>
        """,
        encoding="utf-8",
    )


def test_discover_hls_schedule_reports_finds_common_report_names(tmp_path: Path) -> None:
    _write_xml(tmp_path / "solution1" / "syn" / "report" / "deeplearn_csynth.xml", "dense0", 1, 1)
    (tmp_path / "ignored.txt").write_text("nothing", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "kernel_schedule.rpt").write_text(
        "dense1 target II: 2 achieved II: 3",
        encoding="utf-8",
    )

    reports = discover_hls_schedule_reports(tmp_path)

    assert len(reports) == 2
    assert reports[0].name == "kernel_schedule.rpt"
    assert reports[1].name == "deeplearn_csynth.xml"


def test_summarize_hls_schedule_reports_aggregates_loops_and_layer_matches(tmp_path: Path) -> None:
    _write_xml(tmp_path / "solution1" / "syn" / "report" / "deeplearn_csynth.xml", "dense0_mac_loop", 1, 1)
    _write_xml(tmp_path / "solution2" / "syn" / "report" / "deeplearn_csynth.xml", "dense1_mac_loop", 1, 2)

    summary = summarize_hls_schedule_reports(
        tmp_path,
        requested_by_layer={
            "dense0": 1,
            "dense1": 1,
        },
    )

    assert summary["report_count"] == 2
    assert summary["loop_count"] == 2
    assert summary["failed_loop_count"] == 1
    assert summary["requested_layer_count"] == 2
    assert summary["matched_layer_count"] == 1
    assert summary["failed_layer_count"] == 1
    assert len(summary["comparisons"]) == 2


def test_write_hls_schedule_summary_outputs_json(tmp_path: Path) -> None:
    _write_xml(tmp_path / "solution" / "syn" / "report" / "model_csynth.xml", "dense0_loop", 2, 2)
    out_path = tmp_path / "artifacts" / "hls_schedule_summary.json"

    summary = write_hls_schedule_summary(
        tmp_path,
        out_path,
        requested_by_layer={"dense0": 2},
    )

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["report_count"] == summary["report_count"] == 1
    assert loaded["matched_layer_count"] == 1
