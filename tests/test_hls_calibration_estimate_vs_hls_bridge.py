import json
from pathlib import Path

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset


def test_bridge_uses_estimate_vs_hls_layer_validation_artifact(tmp_path: Path):
    project = tmp_path / "build" / "demo"
    calibration = project / "calibration"
    reports = project / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    layer_validation = project / "estimate_vs_hls" / "layer_validation"
    calibration.mkdir(parents=True)
    reports.mkdir(parents=True)
    layer_validation.mkdir(parents=True)

    compile_plan = calibration / "compile_plan_for_calibration.json"
    compile_plan.write_text(json.dumps({
        "project": "demo",
        "board": "kv260",
        "part": "xck26",
        "clock_target_mhz": 200,
        "layers": [
            {"node_name": "dense0", "op_type": "Dense", "backend_kernel": "dense_out_in_tiled"}
        ],
    }))

    (reports / "dense_out_in_tiled_csynth.rpt").write_text(
        """
        == Utilization Estimates
        |Total| 3 | 2 | 10098 | 5370 | 0 |
        == Performance Estimates
        + Latency:
          min = 14976
          max = 14976
        """
    )

    (layer_validation / "results.json").write_text(json.dumps({
        "layers": [
            {
                "layer_name": "dense0",
                "operator": "Dense",
                "module": "dense_out_in_tiled",
                "predicted": {
                    "lut": 2772,
                    "ff": 5154,
                    "dsp": 2,
                    "bram": 1,
                    "latency_cycles": 6800,
                },
            }
        ]
    }))

    dataset = build_calibration_dataset(compile_plan, project / "hls")
    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["layer_name"] == "dense0"
    assert sample["operator"] == "Dense"
    assert sample["estimated"]["lut"] == 2772
    assert sample["hls_actual"]["lut"] == 5370
    assert sample["hls_actual"]["latency_cycles"] == 14976


def test_bridge_can_reconstruct_prediction_from_error_percent(tmp_path: Path):
    project = tmp_path / "build" / "demo"
    calibration = project / "calibration"
    reports = project / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    layer_validation = project / "estimate_vs_hls" / "layer_validation"
    calibration.mkdir(parents=True)
    reports.mkdir(parents=True)
    layer_validation.mkdir(parents=True)

    compile_plan = calibration / "compile_plan_for_calibration.json"
    compile_plan.write_text(json.dumps({"project": "demo", "layers": [{"node_name": "conv0", "op_type": "Conv"}]}))

    (reports / "conv2d_csynth.rpt").write_text(
        """
        |Total| 0 | 10 | 656 | 673 | 0 |
        + Latency:
          min = 24373
        """
    )

    (layer_validation / "results.json").write_text(json.dumps({
        "rows": [
            {
                "layer_name": "conv0",
                "type": "Conv",
                "module": "conv2d",
                "lut_error": 107.28,
                "ff_error": 245.58,
                "dsp_error": 260.0,
                "latency_cycles_error": 88.81,
            }
        ]
    }))

    dataset = build_calibration_dataset(compile_plan, project / "hls")
    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["layer_name"] == "conv0"
    assert sample["estimated"]["lut"] > 0
    assert sample["hls_actual"]["dsp"] == 10
