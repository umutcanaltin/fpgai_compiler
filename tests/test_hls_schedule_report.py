from __future__ import annotations

from fpgai.analysis.hls_schedule_report import (
    compare_requested_achieved_ii,
    parse_hls_schedule_report,
)


def test_parse_hls_schedule_xml_requested_and_achieved_ii(tmp_path):
    report = tmp_path / "csynth.xml"
    report.write_text(
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
                <TripCountMin>4</TripCountMin>
                <TripCountMax>4</TripCountMax>
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

    parsed = parse_hls_schedule_report(report)

    assert parsed.summary["loop_count"] == 2
    assert parsed.loops[0].name == "dense0_mac_loop"
    assert parsed.loops[0].requested_ii == 2
    assert parsed.loops[0].achieved_ii == 2
    assert parsed.loops[0].ii_met is True
    assert parsed.loops[1].ii_met is False
    assert parsed.summary["failed_ii_count"] == 1


def test_parse_hls_schedule_text_report(tmp_path):
    report = tmp_path / "vitis_hls.log"
    report.write_text(
        "dense0_loop target II = 1 achieved II = 1\n"
        "conv0_loop requested II: 2 final II: 4\n",
        encoding="utf-8",
    )

    parsed = parse_hls_schedule_report(report)

    assert [loop.name for loop in parsed.loops] == [
        "dense0_loop",
        "conv0_loop",
    ]
    assert parsed.loops[1].requested_ii == 2
    assert parsed.loops[1].achieved_ii == 4
    assert parsed.loops[1].ii_met is False


def test_compare_requested_achieved_ii_by_layer(tmp_path):
    report = tmp_path / "csynth.xml"
    report.write_text(
        """
        <profile>
          <Loop><Name>dense0_loop</Name><TargetII>2</TargetII><FinalII>2</FinalII></Loop>
          <Loop><Name>dense1_loop</Name><TargetII>1</TargetII><FinalII>3</FinalII></Loop>
        </profile>
        """,
        encoding="utf-8",
    )

    parsed = parse_hls_schedule_report(report)
    comparison = compare_requested_achieved_ii(
        {"dense0": 2, "dense1": 1, "missing": 1},
        parsed,
    )

    assert comparison["layer_count"] == 3
    assert comparison["matched_layer_count"] == 2
    assert comparison["failed_layer_count"] == 1
    assert comparison["layers"][0]["ii_met"] is True
    assert comparison["layers"][1]["ii_met"] is False
    assert comparison["layers"][2]["ii_met"] is None
