from __future__ import annotations

import json

from fpgai.analysis.hls_calibration_dataset import (
    build_calibration_dataset,
    parse_hls_csynth_report,
)


def test_parse_hls_csynth_report_from_vitis_style_total_row(tmp_path):
    report = tmp_path / "dense_0_csynth.rpt"
    report.write_text(
        """
        + Latency (cycles): min = 5200, max = 5200
        |Total| 3 | 16 | 1300 | 1800 | 0 |
        """
    )

    parsed = parse_hls_csynth_report(report)

    assert parsed["bram"] == 3
    assert parsed["dsp"] == 16
    assert parsed["ff"] == 1300
    assert parsed["lut"] == 1800
    assert parsed["latency_cycles"] == 5200


def test_build_calibration_dataset_matches_layer_report(tmp_path):
    plan = tmp_path / "compile_plan.json"
    plan.write_text(
        json.dumps(
            {
                "project": "toy",
                "board": "kv260",
                "layers": [
                    {
                        "name": "dense_0",
                        "operator": "Dense",
                        "estimated": {
                            "lut": 1000,
                            "ff": 1000,
                            "dsp": 8,
                            "bram": 2,
                            "latency_cycles": 4000,
                        },
                        "features": {"input_size": 128, "output_size": 64},
                    }
                ],
            }
        )
    )
    report_dir = tmp_path / "hls"
    report_dir.mkdir()
    (report_dir / "dense_0_csynth.rpt").write_text(
        """
        Latency (cycles): 5000
        LUT: 1400
        FF: 1200
        DSP: 8
        BRAM_18K: 3
        """
    )

    dataset = build_calibration_dataset(plan, report_dir, tmp_path / "out.json")

    assert dataset["project"] == "toy"
    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["operator"] == "Dense"
    assert sample["estimated"]["lut"] == 1000
    assert sample["hls_actual"]["lut"] == 1400
    assert sample["features"]["input_size"] == 128


def test_build_calibration_dataset_missing_report_is_warning(tmp_path):
    plan = tmp_path / "compile_plan.json"
    plan.write_text(json.dumps({"layers": [{"name": "dense_0", "operator": "Dense", "lut": 1}]}))

    dataset = build_calibration_dataset(plan, tmp_path / "empty")

    assert dataset["samples"] == []
    assert dataset["warnings"][0]["warning"] == "missing_hls_report"
