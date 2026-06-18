from __future__ import annotations

import json

from fpgai.analysis.hls_calibration_model import (
    apply_calibration_model,
    fit_calibration_model,
    mean_absolute_percentage_error,
)
from fpgai.analysis.hls_estimate_report import write_estimate_vs_hls_report


def _toy_dataset():
    return {
        "schema_version": 1,
        "samples": [
            {
                "operator": "Dense",
                "layer_name": "dense_0",
                "estimated": {"lut": 100, "ff": 200, "dsp": 4, "bram": 2, "latency_cycles": 1000},
                "hls_actual": {"lut": 200, "ff": 400, "dsp": 4, "bram": 4, "latency_cycles": 2000},
                "features": {},
            },
            {
                "operator": "Dense",
                "layer_name": "dense_1",
                "estimated": {"lut": 100, "ff": 200, "dsp": 4, "bram": 2, "latency_cycles": 1000},
                "hls_actual": {"lut": 200, "ff": 400, "dsp": 4, "bram": 4, "latency_cycles": 2000},
                "features": {},
            },
        ],
    }


def test_fit_and_apply_calibration_model_reduces_error():
    dataset = _toy_dataset()
    raw_error = mean_absolute_percentage_error(dataset["samples"], "lut")

    model = fit_calibration_model(dataset)
    calibrated = apply_calibration_model(dataset, model)
    calibrated_error = mean_absolute_percentage_error(calibrated["samples"], "lut", calibrated=True)

    assert model["operators"]["Dense"]["lut"] == 2.0
    assert calibrated["samples"][0]["calibrated_estimate"]["lut"] == 200
    assert calibrated_error < raw_error


def test_zero_estimate_nonzero_actual_warns_without_crash():
    dataset = {
        "samples": [
            {
                "operator": "Conv",
                "layer_name": "conv_0",
                "estimated": {"lut": 0, "ff": 1, "dsp": 0, "bram": 0, "latency_cycles": 1},
                "hls_actual": {"lut": 10, "ff": 2, "dsp": 1, "bram": 1, "latency_cycles": 2},
                "features": {},
            }
        ]
    }

    model = fit_calibration_model(dataset)
    calibrated = apply_calibration_model(dataset, model)

    assert model["warnings"]
    assert calibrated["samples"][0]["warnings"]


def test_report_writer_outputs_json_serializable_files(tmp_path):
    dataset = _toy_dataset()
    model = fit_calibration_model(dataset)

    report = write_estimate_vs_hls_report(
        dataset,
        model,
        tmp_path / "estimate_vs_hls.json",
        tmp_path / "summary.txt",
    )

    json.dumps(report)
    assert (tmp_path / "estimate_vs_hls.json").exists()
    assert (tmp_path / "summary.txt").read_text().startswith("FPGAI HLS Calibration Summary")
