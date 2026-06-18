from pathlib import Path
import json

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset


def test_real_artifact_style_nested_estimates_match_primary_module_reports(tmp_path: Path):
    plan = {
        "project": "fpgai_example_dense",
        "some_nested_report": {
            "layer_validation": [
                {
                    "layer_name": "conv0",
                    "operator": "Conv",
                    "predicted_resources": {
                        "lut": 325,
                        "ff": 190,
                        "dsp": 3,
                        "latency_cycles": 2727,
                    },
                },
                {
                    "layer_name": "dense0",
                    "operator": "Dense",
                    "predicted_resources": {
                        "lut": 338,
                        "ff": 410,
                        "dsp": 2,
                        "bram18": 1,
                        "latency_cycles": 6785,
                    },
                },
            ]
        },
    }
    plan_path = tmp_path / "compile_plan_for_calibration.json"
    plan_path.write_text(json.dumps(plan))

    report_dir = tmp_path / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "conv2d_Pipeline_VITIS_LOOP_57_2_csynth.rpt").write_text("|Total| 1 | 99 | 9999 | 9999 | 0 |\nmin = 9999")
    (report_dir / "conv2d_csynth.rpt").write_text("|Total| 0 | 10 | 656 | 673 | 0 |\nmin = 24373")
    (report_dir / "dense_out_in_tiled_csynth.rpt").write_text("|Total| 3 | 2 | 10098 | 5370 | 0 |\nmin = 14976")

    dataset = build_calibration_dataset(plan_path, report_dir)

    assert len(dataset["samples"]) == 2
    conv = next(s for s in dataset["samples"] if s["operator"] == "Conv")
    dense = next(s for s in dataset["samples"] if s["operator"] == "Dense")
    assert conv["hls_report"].endswith("conv2d_csynth.rpt")
    assert conv["hls_actual"]["lut"] == 673
    assert conv["hls_actual"]["latency_cycles"] == 24373
    assert dense["hls_actual"]["bram"] == 3


def test_prefixed_prediction_keys_are_accepted(tmp_path: Path):
    plan = {
        "validation": [
            {
                "layer": "act1",
                "type": "Softmax",
                "pred_lut": 123,
                "pred_ff": 456,
                "pred_dsp": 2,
                "pred_latency_cycles": 77,
            }
        ]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "softmax_typed_10_ap_fixed_s_csynth.rpt").write_text("|Total| 0 | 2 | 3046 | 2732 | 0 |\nmin = 81")

    dataset = build_calibration_dataset(plan_path, report_dir)

    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["operator"] == "Softmax"
    assert sample["estimated"]["lut"] == 123
    assert sample["hls_actual"]["ff"] == 3046
