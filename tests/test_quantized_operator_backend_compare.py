import json
from pathlib import Path

from fpgai.analysis.quantized_operator_backend_compare import build_quantized_add_backend_comparison, write_quantized_add_backend_comparison


def _dump(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quantized_add_backend_comparison_reports_whole_design_deltas(tmp_path: Path) -> None:
    hls_char = _dump(tmp_path / "hls_char.json", {
        "resources": {"lut": 100, "ff": 200, "dsp": 2, "bram": 0, "uram": 0},
        "clock": {"target_mhz": 200.0, "target_period_ns": 5.0, "wns_ns": 0.5, "tns_ns": 0.0, "timing_met": True},
        "power": {"total_power_w": 0.30, "dynamic_power_w": 0.02, "static_power_w": 0.28},
        "artifacts": {},
    })
    vhdl_char = _dump(tmp_path / "vhdl_char.json", {
        "resources": {"lut": 120, "ff": 240, "dsp": 1, "bram": 0, "uram": 0},
        "clock": {"target_mhz": 200.0, "target_period_ns": 5.0, "wns_ns": 0.2, "tns_ns": 0.0, "timing_met": True},
        "power": {"total_power_w": 0.31, "dynamic_power_w": 0.03, "static_power_w": 0.28},
        "artifacts": {},
    })
    hls_tool = _dump(tmp_path / "hls_tool.json", {
        "status": "passed",
        "validation_level": "vivado_implemented",
        "mixed_language_simulation_passed": True,
        "simulation_metrics": {
            "first_output_latency_cycles": 20,
            "packet_completion_latency_cycles": 26,
            "post_input_drain_cycles": 14,
            "input_accept_span_cycles": 6,
            "output_accept_span_cycles": 6,
            "mean_output_interbeat_cycles": 2.0,
            "initiation_interval": None,
            "initiation_interval_status": "not_measured_single_packet_testbench",
        },
    })
    vhdl_tool = _dump(tmp_path / "vhdl_tool.json", {
        "status": "passed",
        "validation_level": "vivado_implemented",
        "mixed_language_simulation_passed": True,
        "simulation_metrics": {
            "first_output_latency_cycles": 24,
            "packet_completion_latency_cycles": 30,
            "post_input_drain_cycles": 18,
            "input_accept_span_cycles": 6,
            "output_accept_span_cycles": 6,
            "mean_output_interbeat_cycles": 2.0,
            "initiation_interval": None,
            "initiation_interval_status": "not_measured_single_packet_testbench",
        },
    })
    hls_phys = _dump(tmp_path / "hls_phys.json", {"numeric_validation": {"comparison": "xsim_exact_word_sequence"}, "edges": []})
    vhdl_phys = _dump(tmp_path / "vhdl_phys.json", {"numeric_validation": {"comparison": "xsim_exact_word_sequence"}, "edges": [{"tensor": "skip_packet", "elastic_buffer_depth_words": 4}]})
    hls_numeric = _dump(tmp_path / "hls_numeric.json", {"schema": "numeric", "input_integer": [1], "expected_integer": [1]})
    vhdl_numeric = _dump(tmp_path / "vhdl_numeric.json", {
        "schema": "numeric",
        "partition": {
            "add": {
                "left_quantization": {"scale": 0.125, "zero_point": 0},
                "right_quantization": {"scale": 0.25, "zero_point": 0},
                "output_quantization": {"scale": 0.5, "zero_point": 0},
                "lowering": {"left_multiplier": 1, "left_shift": 2, "right_multiplier": 1, "right_shift": 1, "rounding_mode": 0, "saturation_mode": 0, "qmin": -128, "qmax": 127},
            }
        },
    })

    payload = build_quantized_add_backend_comparison(
        hls_characterization_path=hls_char,
        hls_tool_result_path=hls_tool,
        hls_physical_path=hls_phys,
        hls_numeric_path=hls_numeric,
        vhdl_characterization_path=vhdl_char,
        vhdl_tool_result_path=vhdl_tool,
        vhdl_physical_path=vhdl_phys,
        vhdl_numeric_path=vhdl_numeric,
    )
    assert payload["status"] == "passed"
    assert payload["delta_vhdl_minus_hls"]["resources"]["lut"]["absolute"] == 20.0
    assert payload["delta_vhdl_minus_hls"]["resources"]["dsp"]["absolute"] == -1.0
    assert payload["variants"]["vhdl"]["buffering"]["max_elastic_buffer_depth_words"] == 4
    assert payload["variants"]["vhdl"]["quantization"]["left_scale"] == 0.125
    assert payload["experimental_design"]["isolated_add_backend_effect"] is False
    assert payload["comparability"]["same_terminal_relu_backend"] is False
    assert payload["comparability"]["latency_comparison_available"] is True
    assert payload["delta_vhdl_minus_hls"]["latency"]["first_output_latency_cycles"]["absolute"] == 4.0
    assert payload["variants"]["hls"]["latency"]["measurement_status"] == "cycle_accurate_behavioral_xsim"

    json_path, md_path = write_quantized_add_backend_comparison(payload, tmp_path / "reports")
    assert json_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "Quantized residual partition backend comparison" in text
    assert "not an isolated Add-only causal comparison" in text
    assert "whole-design routed" in text
    assert "measured Fmax" in text
    assert "First-output latency (cycles)" in text
    assert "Mean output inter-beat spacing (cycles)" in text
