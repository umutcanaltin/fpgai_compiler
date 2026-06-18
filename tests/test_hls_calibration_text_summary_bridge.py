from pathlib import Path

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset


def test_text_summary_bridge_builds_nonzero_samples(tmp_path: Path):
    project = tmp_path / "build" / "fpgai_example_dense"
    calib = project / "calibration"
    calib.mkdir(parents=True)
    hls = project / "hls"
    hls.mkdir()
    (calib / "compile_plan_for_calibration.json").write_text(
        '{"project":"fpgai_example_dense","layers":[{"node_name":"dense0","op_type":"Dense"}]}'
    )

    modules = project / "estimate_vs_hls" / "modules"
    layers = project / "estimate_vs_hls" / "layer_validation"
    modules.mkdir(parents=True)
    layers.mkdir(parents=True)
    (modules / "summary.txt").write_text(
        """
=============== FPGAI HLS Module Breakdown ===============
Primary module                           Type          LUT      FF       DSP   BRAM  Cycles
dense_out_in_tiled                       Dense        5370     10098    2     3     14976
conv2d                                   Conv         673      656      10    0     24373
===========================================================
"""
    )
    (layers / "summary.txt").write_text(
        """
=============== FPGAI Layer vs HLS Validation ===============
Layer                Type       Module                       LUT err   FF err    DSP err   BRAM err  Cycle err
conv0                Conv       conv2d                       107.28%   245.58%   260.00%   n/a       88.81%
dense0               Dense      dense_out_in_tiled           93.71%    95.94%    0.00%     133.33%   54.69%
==========================================================
"""
    )

    out = calib / "hls_operator_dataset.json"
    dataset = build_calibration_dataset(calib / "compile_plan_for_calibration.json", hls, out)

    assert len(dataset["samples"]) == 2
    dense = next(s for s in dataset["samples"] if s["layer_name"] == "dense0")
    assert dense["hls_actual"]["lut"] == 5370
    assert dense["hls_actual"]["ff"] == 10098
    assert dense["hls_actual"]["dsp"] == 2
    assert dense["hls_actual"]["bram"] == 3
    assert dense["hls_actual"]["latency_cycles"] == 14976
    assert dense["estimated"]["lut"] > 0
    assert dense["estimated"]["ff"] > 0
    assert dense["estimated"]["dsp"] == 2
    assert dense["estimated"]["bram"] > 0
    assert dense["estimated"]["latency_cycles"] > 0
    assert any(w.get("warning") == "used_text_summary_bridge" for w in dataset["warnings"])
