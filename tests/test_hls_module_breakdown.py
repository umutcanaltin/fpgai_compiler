from __future__ import annotations

import csv
import json
from pathlib import Path

from fpgai.analysis.hls_module_breakdown import (
    collect_hls_module_reports,
    run_hls_module_breakdown,
)


def _write_report(
    path: Path,
    *,
    module: str,
    lut: int,
    ff: int,
    dsp: int,
    bram: int,
    latency: int,
    interval: int = 1,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        f"""
<Report>
  <UserAssignments>
    <TopModelName>{module}</TopModelName>
  </UserAssignments>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>{bram}</BRAM_18K>
      <DSP>{dsp}</DSP>
      <FF>{ff}</FF>
      <LUT>{lut}</LUT>
    </Resources>
  </AreaEstimates>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Average-caseLatency>{latency}</Average-caseLatency>
      <Interval-min>{interval}</Interval-min>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
</Report>
""",
        encoding="utf-8",
    )


def _create_reports(root: Path) -> None:
    _write_report(
        root / "deeplearn_csynth.xml",
        module="deeplearn",
        lut=18445,
        ff=19822,
        dsp=165,
        bram=30,
        latency=14800,
    )
    _write_report(
        root / "conv2d_0_csynth.xml",
        module="conv2d_0",
        lut=5000,
        ff=6000,
        dsp=90,
        bram=12,
        latency=9000,
    )
    _write_report(
        root / "dense_out_in_0_csynth.xml",
        module="dense_out_in_0",
        lut=3000,
        ff=3500,
        dsp=40,
        bram=8,
        latency=2500,
    )
    _write_report(
        root / "softmax_typed_0_csynth.xml",
        module="softmax_typed_0",
        lut=2200,
        ff=2400,
        dsp=30,
        bram=2,
        latency=400,
    )
    _write_report(
        root / "relu_typed_0_csynth.xml",
        module="relu_typed_0",
        lut=300,
        ff=350,
        dsp=0,
        bram=0,
        latency=100,
    )


def test_collects_and_classifies_module_reports(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    _create_reports(report_dir)

    payload = collect_hls_module_reports(
        report_dir,
        top_name="deeplearn",
    )

    assert payload["available"] is True
    assert payload["format"] == (
        "fpgai.hls_module_breakdown.v2"
    )
    assert payload["module_count"] == 5
    assert payload["primary_module_count"] == 4
    assert payload["helper_module_count"] == 0
    assert payload["child_module_count"] == 4

    top = payload["top_report"]

    assert top is not None
    assert top["module"] == "deeplearn"
    assert top["is_top"] is True
    assert top["hierarchy_role"] == "top"
    assert top["dsp"] == 165
    assert top["latency_cycles"] == 14800

    modules = {
        row["module"]: row
        for row in payload["modules"]
    }

    assert modules["conv2d_0"]["op_type"] == "Conv"
    assert modules["dense_out_in_0"]["op_type"] == "Dense"
    assert modules["softmax_typed_0"]["op_type"] == "Softmax"
    assert modules["relu_typed_0"]["op_type"] == "Relu"

    assert modules["conv2d_0"]["hierarchy_role"] == "primary"
    assert modules["conv2d_0"]["is_generated_helper"] is False


def test_aggregates_primary_resources_by_operator(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    _create_reports(report_dir)

    payload = collect_hls_module_reports(
        report_dir,
        top_name="deeplearn",
    )

    by_operator = payload["by_operator"]

    assert by_operator["Conv"] == {
        "module_count": 1,
        "lut": 5000,
        "ff": 6000,
        "dsp": 90,
        "bram18": 12,
    }
    assert by_operator["Dense"]["dsp"] == 40
    assert by_operator["Softmax"]["dsp"] == 30
    assert by_operator["Relu"]["dsp"] == 0

    expected = {
        "lut": 10500,
        "ff": 12250,
        "dsp": 160,
        "bram18": 22,
    }

    assert payload["primary_sum"] == expected
    assert payload["child_sum"] == expected
    assert payload["helper_sum"] == {
        "lut": 0,
        "ff": 0,
        "dsp": 0,
        "bram18": 0,
    }

    assert payload["unassigned_top_resources"] == {
        "lut": 7945,
        "ff": 7572,
        "dsp": 5,
        "bram18": 8,
    }


def test_generated_pipeline_helpers_are_not_double_counted(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"

    _write_report(
        report_dir / "deeplearn_csynth.xml",
        module="deeplearn",
        lut=18445,
        ff=19822,
        dsp=165,
        bram=30,
        latency=14800,
    )
    _write_report(
        report_dir / "conv2d_csynth.xml",
        module="conv2d",
        lut=10385,
        ff=14264,
        dsp=154,
        bram=0,
        latency=3045,
    )
    _write_report(
        report_dir
        / "conv2d_Pipeline_VITIS_LOOP_48_2_csynth.xml",
        module="conv2d_Pipeline_VITIS_LOOP_48_2",
        lut=10208,
        ff=14080,
        dsp=154,
        bram=0,
        latency=758,
    )
    _write_report(
        report_dir / "dense_out_in_csynth.xml",
        module="dense_out_in",
        lut=416,
        ff=624,
        dsp=2,
        bram=3,
        latency=6768,
    )

    payload = collect_hls_module_reports(
        report_dir,
        top_name="deeplearn",
    )

    assert payload["primary_module_count"] == 2
    assert payload["helper_module_count"] == 1

    assert payload["by_operator"]["Conv"] == {
        "module_count": 1,
        "lut": 10385,
        "ff": 14264,
        "dsp": 154,
        "bram18": 0,
    }

    assert payload["helper_by_operator"]["Conv"] == {
        "module_count": 1,
        "lut": 10208,
        "ff": 14080,
        "dsp": 154,
        "bram18": 0,
    }

    assert payload["primary_sum"]["dsp"] == 156
    assert payload["helper_sum"]["dsp"] == 154

    primary_names = {
        row["module"]
        for row in payload["primary_modules"]
    }
    helper_names = {
        row["module"]
        for row in payload["helper_modules"]
    }

    assert primary_names == {
        "conv2d",
        "dense_out_in",
    }
    assert helper_names == {
        "conv2d_Pipeline_VITIS_LOOP_48_2",
    }


def test_discovers_reports_recursively(
    tmp_path: Path,
) -> None:
    report_dir = (
        tmp_path
        / "solution"
        / "syn"
        / "report"
    )

    _write_report(
        report_dir / "deeplearn_csynth.xml",
        module="deeplearn",
        lut=1000,
        ff=1200,
        dsp=10,
        bram=4,
        latency=500,
    )
    _write_report(
        report_dir
        / "nested"
        / "conv2d_csynth.xml",
        module="conv2d",
        lut=500,
        ff=600,
        dsp=8,
        bram=2,
        latency=300,
    )

    payload = collect_hls_module_reports(
        tmp_path,
        top_name="deeplearn",
    )

    assert payload["module_count"] == 2
    assert payload["top_report"]["module"] == "deeplearn"
    assert payload["by_operator"]["Conv"]["module_count"] == 1


def test_duplicate_module_reports_use_larger_result(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"

    _write_report(
        report_dir
        / "first"
        / "conv2d_csynth.xml",
        module="conv2d",
        lut=100,
        ff=100,
        dsp=1,
        bram=1,
        latency=100,
    )
    _write_report(
        report_dir
        / "second"
        / "conv2d_csynth.xml",
        module="conv2d",
        lut=500,
        ff=600,
        dsp=8,
        bram=2,
        latency=300,
    )

    payload = collect_hls_module_reports(
        report_dir,
    )

    assert payload["module_count"] == 1

    module = payload["modules"][0]

    assert module["lut"] == 500
    assert module["ff"] == 600
    assert module["dsp"] == 8
    assert module["bram18"] == 2


def test_writes_module_breakdown_artifacts(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    output_dir = tmp_path / "build"

    _create_reports(report_dir)

    result = run_hls_module_breakdown(
        out_dir=output_dir,
        report_path=report_dir,
        top_name="deeplearn",
    )

    assert result.available is True
    assert result.results_json.is_file()
    assert result.results_csv.is_file()
    assert result.summary_txt.is_file()

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["top_report"]["module"] == "deeplearn"
    assert payload["by_operator"]["Softmax"]["dsp"] == 30

    with result.results_csv.open(
        newline="",
        encoding="utf-8",
    ) as input_file:
        rows = list(
            csv.DictReader(input_file)
        )

    assert len(rows) == 5
    assert {
        row["module"]
        for row in rows
    } == {
        "deeplearn",
        "conv2d_0",
        "dense_out_in_0",
        "softmax_typed_0",
        "relu_typed_0",
    }

    assert "FPGAI HLS Module Breakdown" in (
        result.terminal_summary
    )
    assert "Primary resources by operator" in (
        result.terminal_summary
    )
    assert "Generated helper reports are" in (
        result.terminal_summary
    )


def test_missing_reports_produce_valid_empty_artifacts(
    tmp_path: Path,
) -> None:
    result = run_hls_module_breakdown(
        out_dir=tmp_path / "build",
        report_path=tmp_path / "missing",
        top_name="deeplearn",
    )

    assert result.available is False
    assert result.results_json.is_file()
    assert result.results_csv.is_file()
    assert result.summary_txt.is_file()

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8",
        )
    )

    assert payload["available"] is False
    assert payload["module_count"] == 0
    assert payload["primary_module_count"] == 0
    assert payload["helper_module_count"] == 0
    assert payload["modules"] == []