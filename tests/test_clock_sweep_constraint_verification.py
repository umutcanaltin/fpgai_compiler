from pathlib import Path

from fpgai.analysis.clock_sweep import point_from_characterization, summarize_clock_sweep


def _write_timing(path: Path, period: float) -> None:
    path.write_text(
        f"Clock Name:         clk_pl_0\n\nPeriod(ns):         {period:.3f}\n",
        encoding="utf-8",
    )


def test_clock_sweep_rejects_unpropagated_vivado_clock(tmp_path):
    timing = tmp_path / "timing_impl.rpt"
    _write_timing(timing, 10.0)
    point = point_from_characterization(
        300.0,
        {"status": "passed", "clock": {"timing_met": True, "wns_ns": 5.5}},
        timing_report=timing,
        hls_payload={"clock": {"target_period_ns": 3.333333}},
    )
    assert point.constraint_verified is False
    assert point.status == "constraint_mismatch"
    assert point.vivado_clock_mhz == 100.0
    report = summarize_clock_sweep([point])
    assert report["highest_passing_mhz"] is None
    assert report["status"] == "constraint_unverified"


def test_clock_sweep_accepts_actual_requested_vivado_clock(tmp_path):
    timing = tmp_path / "timing_impl.rpt"
    _write_timing(timing, 4.0)
    point = point_from_characterization(
        250.0,
        {"status": "passed", "clock": {"timing_met": True, "wns_ns": 0.2}},
        timing_report=timing,
        hls_payload={"clock": {"target_period_ns": 4.0}},
    )
    assert point.constraint_verified is True
    assert point.status == "passed"
    assert point.vivado_clock_mhz == 250.0
    report = summarize_clock_sweep([point])
    assert report["highest_passing_mhz"] == 250.0
