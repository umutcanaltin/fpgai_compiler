from __future__ import annotations

import json
from pathlib import Path

from fpgai.engine.compiler import Compiler


def test_compiler_emits_hls_schedule_summary(tmp_path: Path) -> None:
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
                <LatencyMin>10</LatencyMin>
                <LatencyMax>12</LatencyMax>
              </Loop>
            </SummaryOfLoopLatency>
          </PerformanceEstimates>
        </profile>
        """,
        encoding="utf-8",
    )

    result = Compiler._emit_hls_schedule_summary(
        None,
        tmp_path,
    )

    assert result is not None
    assert result["path"] == "hls_schedule_summary.json"

    summary_path = tmp_path / "hls_schedule_summary.json"
    assert summary_path.exists()

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["summary"]["report_count"] == 1
    assert data["summary"]["loop_count"] == 1


def test_compiler_source_records_schedule_summary_in_manifest() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "write_hls_schedule_summary" in source
    assert "hls_schedule_summary.json" in source
    assert 'manifest["hls_schedule_summary"] = hls_schedule_summary' in source
