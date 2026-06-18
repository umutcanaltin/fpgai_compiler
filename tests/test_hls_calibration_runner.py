from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from fpgai.analysis.hls_calibration_runner import run_hls_calibration


@dataclass
class DummyLayerPlan:
    name: str
    operator: str
    estimated: dict
    features: dict


@dataclass
class DummyCompilePlan:
    layer_plans: list[DummyLayerPlan]


def test_runner_materializes_plan_and_writes_outputs(tmp_path: Path):
    out_dir = tmp_path / "build"
    report_dir = out_dir / "hls" / "reports"
    report_dir.mkdir(parents=True)

    (report_dir / "dense_0_csynth.rpt").write_text(
        """
        == Utilization Estimates
        |Total| 3 | 8 | 1200 | 1500 | 0 |
        == Performance Estimates
        Latency (cycles): 5200
        """,
        encoding="utf-8",
    )

    plan = DummyCompilePlan(
        layer_plans=[
            DummyLayerPlan(
                name="dense_0",
                operator="Dense",
                estimated={"lut": 1000, "ff": 800, "dsp": 8, "bram": 2, "latency_cycles": 4000},
                features={"input_size": 784, "output_size": 64, "precision_bits": 16},
            )
        ]
    )

    result = run_hls_calibration(
        out_dir=out_dir,
        raw_cfg={"analysis": {"hls_calibration": {"enabled": True}}},
        compile_plan=plan,
        hls_report_dir=out_dir,
        clock_mhz=200.0,
    )

    assert not result.skipped
    assert result.sample_count == 1
    assert result.dataset_json is not None and result.dataset_json.exists()
    assert result.calibrated_model_json is not None and result.calibrated_model_json.exists()
    assert result.estimate_vs_hls_json is not None and result.estimate_vs_hls_json.exists()
    assert result.summary_txt is not None and result.summary_txt.exists()

    payload = json.loads(result.estimate_vs_hls_json.read_text(encoding="utf-8"))
    assert payload["summary"]["sample_count"] == 1


def test_runner_skips_when_disabled(tmp_path: Path):
    result = run_hls_calibration(
        out_dir=tmp_path / "build",
        raw_cfg={"analysis": {"hls_calibration": {"enabled": False}}},
    )
    assert result.skipped
    assert (result.out_dir / "skipped.txt").exists()
