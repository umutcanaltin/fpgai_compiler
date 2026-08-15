import json
from pathlib import Path

from fpgai.analysis.hls_synthesis_characterization import (
    characterize_hls_synthesis,
    write_hls_synthesis_characterization,
)


def test_characterizes_vitis_csynth_xml(tmp_path: Path) -> None:
    report = tmp_path / "deeplearn_csynth.xml"
    report.write_text(
        """<Report>
<AreaEstimates><Resources><BRAM_18K>12</BRAM_18K><DSP>7</DSP><FF>345</FF><LUT>678</LUT><URAM>3</URAM></Resources></AreaEstimates>
<PerformanceEstimates>
<SummaryOfOverallLatency><Best-caseLatency>100</Best-caseLatency><Worst-caseLatency>120</Worst-caseLatency><Interval-min>4</Interval-min><Interval-max>5</Interval-max></SummaryOfOverallLatency>
<SummaryOfTimingAnalysis><EstimatedClockPeriod>4.5</EstimatedClockPeriod></SummaryOfTimingAnalysis>
</PerformanceEstimates></Report>""",
        encoding="utf-8",
    )
    result = characterize_hls_synthesis(
        csynth_report_path=report,
        target_clock_mhz=200.0,
        top_name="deeplearn",
        participating_external_packages=("community.scale_bias_hls",),
        declared_implementation_metrics={"scale_bias_0": {"latency_cycles": 32}},
    )
    assert result.status == "passed"
    assert result.resources == {"lut": 678, "ff": 345, "dsp": 7, "bram18": 12, "uram": 3}
    assert result.latency_min_cycles == 100
    assert result.latency_max_cycles == 120
    assert result.initiation_interval_min == 4
    assert result.initiation_interval_max == 5
    assert result.estimated_clock_period_ns == 4.5
    assert result.estimated_fmax_mhz == 1000.0 / 4.5
    assert result.timing_margin_ns == 0.5
    assert result.target_met is True
    assert result.measurement_comparability == "whole_design_only"


def test_characterization_reports_missing_or_not_run() -> None:
    not_run = characterize_hls_synthesis(
        csynth_report_path=None, target_clock_mhz=200, top_name="deeplearn"
    )
    assert not_run.status == "not_run"
    assert not_run.target_met is None


def test_writes_reproducible_characterization_reports(tmp_path: Path) -> None:
    report = tmp_path / "csynth.rpt"
    report.write_text(
        "Estimated Clock Period: 4.75\nLatency (cycles): min = 90, max = 110\nInterval: min = 2, max = 3\n",
        encoding="utf-8",
    )
    result = characterize_hls_synthesis(
        csynth_report_path=report,
        target_clock_mhz=200,
        top_name="deeplearn",
    )
    json_path, md_path = write_hls_synthesis_characterization(result, tmp_path / "reports")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fpgai.hls-synthesis-characterization/v1"
    assert payload["scope"] == "mixed_graph_top"
    assert payload["validation_level"] == "hls_synthesized"
    assert "whole-design HLS synthesis measurements" in md_path.read_text(encoding="utf-8")
