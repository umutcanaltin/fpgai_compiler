from __future__ import annotations

import json
from pathlib import Path

from fpgai.analysis.hls_estimate_compare import (
    parse_hls_csynth_report,
    run_estimate_vs_hls_compare,
)
from fpgai.analysis.performance_estimator import (
    estimate_performance,
)
from fpgai.analysis.resource_estimator import (
    estimate_resources_from_descriptors,
)
from fpgai.engine.models import LayerDescriptor


def _spec(total_bits: int, int_bits: int) -> dict[str, object]:
    return {
        "type": "ap_fixed",
        "total_bits": total_bits,
        "int_bits": int_bits,
    }


def _config(
    *,
    activation_bits: int = 16,
    weight_bits: int = 16,
    pe: int = 1,
    simd: int = 1,
) -> dict:
    return {
        "targets": {
            "platform": {
                "clocks": [
                    {
                        "name": "pl_clk0",
                        "target_mhz": 200,
                    }
                ]
            }
        },
        "numerics": {
            "defaults": {
                "activation": _spec(
                    activation_bits,
                    min(6, activation_bits),
                ),
                "weight": _spec(
                    weight_bits,
                    min(6, weight_bits),
                ),
                "bias": _spec(24, 10),
                "accum": _spec(24, 10),
            },
            "layers": [],
        },
        "optimization": {
            "parallel": {
                "pe": pe,
                "simd": simd,
                "unroll_factor": 1,
                "partition_factor": 1,
                "pipeline_ii": 1,
            }
        },
        "memory": {
            "storage": {
                "weights": "bram",
                "activations": "bram",
            }
        },
        "analysis": {
            "design_space": {
                "performance": {
                    "baseline_cpu_latency_ms": 1.6,
                },
                "estimator": {
                    "minimum_bram_elements": 512,
                    "input_words_per_cycle": 1.0,
                    "output_words_per_cycle": 1.0,
                },
            }
        },
    }


def _dense_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        input_shapes=[
            (1, 128),
            (64, 128),
            (64,),
        ],
        output_shapes=[(1, 64)],
        param_names=["weights", "bias"],
        param_bytes=64 * 128 * 4 + 64 * 4,
        activation_bytes_in=128 * 4,
        activation_bytes_out=64 * 4,
        macs=128 * 64,
        attrs={
            "in_features": 128,
            "out_features": 64,
        },
        compute_hint="compute_bound",
        backend_kernel="dense",
    )


def _conv_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="conv0",
        op_type="Conv",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        input_shapes=[
            (1, 3, 28, 28),
            (16, 3, 3, 3),
            (16,),
        ],
        output_shapes=[(1, 16, 26, 26)],
        param_names=["weights", "bias"],
        param_bytes=16 * 3 * 3 * 3 * 4 + 16 * 4,
        activation_bytes_in=3 * 28 * 28 * 4,
        activation_bytes_out=16 * 26 * 26 * 4,
        macs=16 * 26 * 26 * 3 * 3 * 3,
        attrs={
            "kernel_shape": [3, 3],
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "out_channels": 16,
        },
        compute_hint="compute_bound",
        backend_kernel="conv",
    )


def _relu_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="relu0",
        op_type="Relu",
        inputs=["input"],
        outputs=["output"],
        input_shapes=[(1, 64)],
        output_shapes=[(1, 64)],
        param_names=[],
        param_bytes=0,
        activation_bytes_in=64 * 4,
        activation_bytes_out=64 * 4,
        macs=0,
        attrs={},
        compute_hint="memory_bound",
        backend_kernel="relu",
    )


def test_resource_estimator_defaults_to_analytical_mode() -> None:
    result = estimate_resources_from_descriptors(
        [_dense_descriptor(), _relu_descriptor()],
        _config(),
    )

    assert result["estimation_mode"] == "analytical"
    assert result["analytical_model"] == "operator_structural_v2"
    assert result["calibration"]["enabled"] is False
    assert len(result["layers"]) == 2

    totals = result["totals"]

    assert totals["predicted_lut"] > 0
    assert totals["predicted_ff"] > 0
    assert totals["predicted_dsp"] > 0
    assert totals["predicted_bram18"] > 0
    assert totals["total_macs"] == 128 * 64


def test_analytical_totals_equal_components_without_calibration() -> None:
    result = estimate_resources_from_descriptors(
        [_dense_descriptor(), _relu_descriptor()],
        _config(),
    )

    totals = result["totals"]
    top = result["top_level"]
    layers = result["layers"]

    assert totals["predicted_lut"] == (
        top["predicted_lut"]
        + sum(layer["predicted_lut"] for layer in layers)
    )
    assert totals["predicted_ff"] == (
        top["predicted_ff"]
        + sum(layer["predicted_ff"] for layer in layers)
    )
    assert totals["predicted_dsp"] == (
        top["predicted_dsp"]
        + sum(layer["predicted_dsp"] for layer in layers)
    )
    assert totals["predicted_bram18"] == (
        top["predicted_bram18"]
        + sum(layer["predicted_bram18"] for layer in layers)
    )


def test_dense_dimensions_and_components_are_reported() -> None:
    result = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        _config(),
    )

    layer = result["layers"][0]
    dimensions = layer["dimensions"]
    components = layer["resource_components"]

    assert dimensions["input_features"] == 128
    assert dimensions["output_features"] == 64
    assert dimensions["weight_elements"] == 128 * 64
    assert dimensions["macs"] == 128 * 64

    assert components["arithmetic_lut"] > 0
    assert components["arithmetic_ff"] > 0
    assert components["arithmetic_dsp"] > 0
    assert components["parameter_bram18"] > 0


def test_conv_dimensions_and_line_buffer_are_reported() -> None:
    result = estimate_resources_from_descriptors(
        [_conv_descriptor()],
        _config(),
    )

    layer = result["layers"][0]
    dimensions = layer["dimensions"]

    assert dimensions["input_channels"] == 3
    assert dimensions["output_channels"] == 16
    assert dimensions["input_height"] == 28
    assert dimensions["input_width"] == 28
    assert dimensions["output_height"] == 26
    assert dimensions["output_width"] == 26
    assert dimensions["kernel_height"] == 3
    assert dimensions["kernel_width"] == 3

    assert layer["resource_components"][
        "line_buffer_bram18"
    ] > 0


def test_more_parallelism_increases_structural_resources() -> None:
    descriptor = _dense_descriptor()

    serial = estimate_resources_from_descriptors(
        [descriptor],
        _config(pe=1, simd=1),
    )
    parallel = estimate_resources_from_descriptors(
        [descriptor],
        _config(pe=4, simd=4),
    )

    assert serial["layers"][0]["multiplier_lanes"] == 1
    assert parallel["layers"][0]["multiplier_lanes"] == 16

    assert (
        parallel["totals"]["predicted_lut"]
        > serial["totals"]["predicted_lut"]
    )
    assert (
        parallel["totals"]["predicted_ff"]
        > serial["totals"]["predicted_ff"]
    )
    assert (
        parallel["totals"]["predicted_dsp"]
        > serial["totals"]["predicted_dsp"]
    )


def test_wider_precision_increases_structural_cost() -> None:
    narrow = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        _config(
            activation_bits=8,
            weight_bits=8,
        ),
    )
    wide = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        _config(
            activation_bits=32,
            weight_bits=32,
        ),
    )

    assert (
        wide["totals"]["predicted_lut"]
        > narrow["totals"]["predicted_lut"]
    )
    assert (
        wide["totals"]["predicted_ff"]
        > narrow["totals"]["predicted_ff"]
    )
    assert (
        wide["totals"]["predicted_dsp"]
        > narrow["totals"]["predicted_dsp"]
    )


def test_calibration_is_only_applied_when_explicitly_configured() -> None:
    config = _config()

    baseline = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        config,
    )

    config["analysis"]["design_space"]["calibration"] = {
        "resources": {
            "enabled": True,
            "fixed_lut": 0,
            "lut_scale": 2.0,
        }
    }

    calibrated = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        config,
    )

    assert calibrated["estimation_mode"] == (
        "analytical_with_explicit_calibration"
    )
    assert calibrated["calibration"]["enabled"] is True
    assert calibrated["totals"]["predicted_lut"] == (
        baseline["totals"]["predicted_lut"] * 2
    )


def test_performance_defaults_to_analytical_schedule() -> None:
    config = _config()
    resources = estimate_resources_from_descriptors(
        [_dense_descriptor(), _relu_descriptor()],
        config,
    )

    result = estimate_performance(
        resource_estimate=resources,
        raw_cfg=config,
    )

    assert result["estimation_mode"] == "analytical"
    assert result["analytical_performance_model"] == (
        "operator_execution_schedule_v2"
    )
    assert result["calibration"]["enabled"] is False
    assert result["predicted_fixed_cycles"] == 0
    assert result["predicted_compute_cycles"] > 0
    assert result["predicted_transfer_cycles"] > 0
    assert result["predicted_control_cycles"] > 0
    assert result["predicted_cycles"] == (
        result["predicted_analytical_cycles"]
    )
    assert len(result["layer_cycles"]) == 2


def test_more_parallelism_reduces_compute_cycles() -> None:
    descriptor = _dense_descriptor()

    serial_config = _config(pe=1, simd=1)
    parallel_config = _config(pe=4, simd=4)

    serial = estimate_performance(
        resource_estimate=estimate_resources_from_descriptors(
            [descriptor],
            serial_config,
        ),
        raw_cfg=serial_config,
    )
    parallel = estimate_performance(
        resource_estimate=estimate_resources_from_descriptors(
            [descriptor],
            parallel_config,
        ),
        raw_cfg=parallel_config,
    )

    assert (
        parallel["predicted_compute_cycles"]
        < serial["predicted_compute_cycles"]
    )


def test_explicit_performance_calibration_remains_supported() -> None:
    config = _config()
    resources = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        config,
    )

    baseline = estimate_performance(
        resource_estimate=resources,
        raw_cfg=config,
    )

    config["analysis"]["design_space"]["calibration"] = {
        "performance": {
            "enabled": True,
            "fixed_cycles": 100,
            "cycle_scale": 2.0,
        }
    }

    calibrated = estimate_performance(
        resource_estimate=resources,
        raw_cfg=config,
    )

    assert calibrated["estimation_mode"] == (
        "analytical_with_explicit_calibration"
    )
    assert calibrated["predicted_fixed_cycles"] == 100
    assert calibrated["predicted_cycles"] == (
        baseline["predicted_cycles"] + 100
    ) * 2


def test_parse_hls_xml_report(tmp_path: Path) -> None:
    report = tmp_path / "deeplearn_csynth.xml"

    report.write_text(
        """
<Report>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>30</BRAM_18K>
      <DSP>165</DSP>
      <FF>19822</FF>
      <LUT>18445</LUT>
    </Resources>
  </AreaEstimates>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Average-caseLatency>14800</Average-caseLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
</Report>
""",
        encoding="utf-8",
    )

    parsed = parse_hls_csynth_report(report)

    assert parsed["actual_lut"] == 18445
    assert parsed["actual_ff"] == 19822
    assert parsed["actual_dsp"] == 165
    assert parsed["actual_bram18"] == 30
    assert parsed["actual_latency_cycles"] == 14800


def test_comparison_reports_model_diagnostics(
    tmp_path: Path,
) -> None:
    report = tmp_path / "deeplearn_csynth.xml"

    report.write_text(
        """
<Report>
  <LUT>18000</LUT>
  <FF>20000</FF>
  <DSP>160</DSP>
  <BRAM_18K>32</BRAM_18K>
  <Average-caseLatency>15000</Average-caseLatency>
</Report>
""",
        encoding="utf-8",
    )

    result = run_estimate_vs_hls_compare(
        out_dir=tmp_path,
        design_space_summary={
            "predicted_lut": 15000,
            "predicted_ff": 16000,
            "predicted_dsp": 40,
            "predicted_bram18": 30,
            "predicted_cycles": 12000.0,
            "predicted_latency_ms": 0.06,
        },
        csynth_report_path=report,
        clock_mhz=200.0,
    )

    payload = json.loads(
        result.results_json.read_text(encoding="utf-8")
    )

    assert payload["available"] is True
    assert payload["comparison_model"] == (
        "analytical_validation_v2"
    )
    assert payload["model_diagnostics"][
        "requires_model_revision"
    ] is True
    assert "dsp" in payload["model_diagnostics"][
        "poor_fields"
    ]
    assert payload["calibration_recommendation"][
        "deprecated"
    ] is True
    assert "Analytical model revision required" in (
        result.terminal_summary
    )


def test_comparison_handles_missing_report(
    tmp_path: Path,
) -> None:
    result = run_estimate_vs_hls_compare(
        out_dir=tmp_path,
        design_space_summary={
            "predicted_lut": 100,
        },
        csynth_report_path=tmp_path / "missing.xml",
        clock_mhz=200.0,
    )

    payload = json.loads(
        result.results_json.read_text(encoding="utf-8")
    )

    assert payload["available"] is False
    assert payload["actual"] is None
    assert payload["model_diagnostics"] is None