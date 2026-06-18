import json
from pathlib import Path

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset, parse_hls_csynth_report


def test_invalid_xml_falls_back_to_sibling_rpt(tmp_path: Path):
    report = tmp_path / "relu_typed_csynth.rpt"
    xml = tmp_path / "relu_typed_csynth.xml"
    report.write_text("""
    |Total| 0 | 0 | 54 | 221 | 0 |
    + Latency:
      min = 2706
    """)
    xml.write_text("this is not valid xml < broken")

    parsed = parse_hls_csynth_report(xml)
    assert parsed["lut"] == 221
    assert parsed["ff"] == 54
    assert parsed["latency_cycles"] == 2706


def test_bridge_uses_modules_results_for_actual_and_layer_errors_for_prediction(tmp_path: Path):
    project = tmp_path / "build" / "demo"
    calibration = project / "calibration"
    reports = project / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    layer_validation = project / "estimate_vs_hls" / "layer_validation"
    modules = project / "estimate_vs_hls" / "modules"
    calibration.mkdir(parents=True)
    reports.mkdir(parents=True)
    layer_validation.mkdir(parents=True)
    modules.mkdir(parents=True)

    compile_plan = calibration / "compile_plan_for_calibration.json"
    compile_plan.write_text(json.dumps({
        "project": "demo",
        "board": "kv260",
        "clock_target_mhz": 200,
        "layers": [{"node_name": "conv0", "op_type": "Conv"}],
    }))

    # The bad XML should not break the bridge because modules/results.json has actuals.
    (reports / "conv2d_csynth.xml").write_text("not valid xml < broken")
    (reports / "conv2d_csynth.rpt").write_text("""
    |Total| 0 | 10 | 656 | 673 | 0 |
    + Latency:
      min = 24373
    """)

    (modules / "results.json").write_text(json.dumps({
        "modules": [
            {
                "module": "conv2d",
                "type": "Conv",
                "actual": {"lut": 673, "ff": 656, "dsp": 10, "bram": 0, "latency_cycles": 24373},
            }
        ]
    }))
    (layer_validation / "results.json").write_text(json.dumps({
        "rows": [
            {
                "layer_name": "conv0",
                "type": "Conv",
                "module": {"name": "conv2d"},
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
    assert sample["operator"] == "Conv"
    assert sample["hls_actual"]["lut"] == 673
    assert sample["hls_actual"]["latency_cycles"] == 24373
    assert sample["estimated"]["lut"] > 0
