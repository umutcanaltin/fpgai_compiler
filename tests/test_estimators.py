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


def _spec(
    total_bits: int,
    int_bits: int,
) -> dict[str, object]:
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
            }
        },
        "analysis": {
            "design_space": {
                "performance": {
                    "baseline_cpu_latency_ms": 1.6,
                },
                "calibration": {
                    "resources": {
                        "fixed_lut": 7500,
                        "fixed_ff": 8000,
                        "fixed_dsp": 14,
                        "fixed_bram18": 2,
                        "lut_scale": 1.0,
                        "ff_scale": 1.0,
                        "dsp_scale": 1.0,
                        "bram18_scale": 1.0,
                    },
                    "performance": {
                        "fixed_cycles": 2000,
                        "cycle_scale": 1.0,
                        "input_words_per_cycle": 1.0,
                        "output_words_per_cycle": 1.0,
                    },
                },
            }
        },
    }


def _dense_descriptor() -> LayerDescriptor:
    return LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=[
            "input",
            "weights",
            "bias",
        ],
        outputs=[
            "output",
        ],
        input_shapes=[
            (1, 128),
            (64, 128),
            (64,),
        ],
        output_shapes=[
            (1, 64),
        ],
        param_names=[
            "weights",
            "bias",
        ],
        param_bytes=(
            64 * 128 * 4
            + 64 * 4
        ),
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
        inputs=[
            "input",
            "weights",
            "bias",
        ],
        outputs=[
            "output",
        ],
        input_shapes=[
            (1, 3, 28, 28),
            (16, 3, 3, 3),
            (16,),
        ],
        output_shapes=[
            (1, 16, 26, 26),
        ],
        param_names=[
            "weights",
            "bias",
        ],
        param_bytes=(
            16 * 3 * 3 * 3 * 4
            + 16 * 4
        ),
        activation_bytes_in=(
            3 * 28 * 28 * 4
        ),
        activation_bytes_out=(
            16 * 26 * 26 * 4
        ),
        macs=(
            16
            * 26
            * 26
            * 3
            * 3
            * 3
        ),
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
        input_shapes=[
            (1, 64),
        ],
        output_shapes=[
            (1, 64),
        ],
        param_names=[],
        param_bytes=0,
        activation_bytes_in=64 * 4,
        activation_bytes_out=64 * 4,
        macs=0,
        attrs={},
        compute_hint="memory_bound",
        backend_kernel="relu",
    )


def test_resource_estimator_returns_compatible_totals() -> None:
    result = estimate_resources_from_descriptors(
        [
            _dense_descriptor(),
            _relu_descriptor(),
        ],
        _config(),
    )

    totals = result["totals"]

    assert result["model"] == (
        "operator_aware_calibrated_v1"
    )
    assert totals["predicted_lut"] > 0
    assert totals["predicted_ff"] > 0
    assert totals["predicted_dsp"] > 0
    assert totals["predicted_bram18"] > 0
    assert totals["total_macs"] == 128 * 64
    assert len(result["layers"]) == 2


def test_dense_dimensions_are_reported() -> None:
    result = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        _config(),
    )

    layer = result["layers"][0]

    assert layer["op_type"] == "Dense"
    assert layer["dimensions"][
        "input_features"
    ] == 128
    assert layer["dimensions"][
        "output_features"
    ] == 64
    assert layer["dimensions"]["macs"] == (
        128 * 64
    )


def test_conv_dimensions_are_reported() -> None:
    result = estimate_resources_from_descriptors(
        [_conv_descriptor()],
        _config(),
    )

    dimensions = result["layers"][0][
        "dimensions"
    ]

    assert dimensions["input_channels"] == 3
    assert dimensions["output_channels"] == 16
    assert dimensions["input_height"] == 28
    assert dimensions["input_width"] == 28
    assert dimensions["output_height"] == 26
    assert dimensions["output_width"] == 26
    assert dimensions["kernel_height"] == 3
    assert dimensions["kernel_width"] == 3


def test_more_parallelism_increases_resources() -> None:
    descriptor = _dense_descriptor()

    serial = estimate_resources_from_descriptors(
        [descriptor],
        _config(
            pe=1,
            simd=1,
        ),
    )
    parallel = estimate_resources_from_descriptors(
        [descriptor],
        _config(
            pe=4,
            simd=4,
        ),
    )

    serial_layer = serial["layers"][0]
    parallel_layer = parallel["layers"][0]

    assert serial_layer[
        "multiplier_lanes"
    ] == 1
    assert parallel_layer[
        "multiplier_lanes"
    ] == 16

    assert parallel["totals"][
        "predicted_lut"
    ] > serial["totals"]["predicted_lut"]

    assert parallel["totals"][
        "predicted_ff"
    ] > serial["totals"]["predicted_ff"]

    assert parallel["totals"][
        "predicted_dsp"
    ] > serial["totals"]["predicted_dsp"]


def test_wider_precision_increases_logic_cost() -> None:
    descriptor = _dense_descriptor()

    narrow = estimate_resources_from_descriptors(
        [descriptor],
        _config(
            activation_bits=8,
            weight_bits=8,
        ),
    )
    wide = estimate_resources_from_descriptors(
        [descriptor],
        _config(
            activation_bits=24,
            weight_bits=24,
        ),
    )

    assert wide["totals"][
        "predicted_lut"
    ] > narrow["totals"]["predicted_lut"]

    assert wide["totals"][
        "predicted_ff"
    ] > narrow["totals"]["predicted_ff"]

    assert wide["totals"][
        "predicted_dsp"
    ] > narrow["totals"]["predicted_dsp"]


def test_resource_calibration_scales_totals() -> None:
    config = _config()

    baseline = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        config,
    )

    config["analysis"]["design_space"][
        "calibration"
    ]["resources"]["lut_scale"] = 2.0

    scaled = estimate_resources_from_descriptors(
        [_dense_descriptor()],
        config,
    )

    assert scaled["totals"]["predicted_lut"] == (
        baseline["totals"]["predicted_lut"]
        * 2
    )


def test_performance_estimator_returns_cycle_breakdown() -> None:
    resources = (
        estimate_resources_from_descriptors(
            [
                _dense_descriptor(),
                _relu_descriptor(),
            ],
            _config(),
        )
    )

    performance = estimate_performance(
        resource_estimate=resources,
        raw_cfg=_config(),
    )

    assert performance["performance_model"] == (
        "operator_schedule_calibrated_v1"
    )
    assert performance[
        "predicted_compute_cycles"
    ] > 0
    assert performance[
        "predicted_transfer_cycles"
    ] > 0
    assert performance[
        "predicted_fixed_cycles"
    ] == 2000
    assert performance["predicted_cycles"] > 2000
    assert performance[
        "predicted_latency_ms"
    ] > 0
    assert performance[
        "predicted_throughput_fps"
    ] > 0
    assert len(
        performance["layer_cycles"]
    ) == 2


def test_more_parallelism_reduces_compute_cycles() -> None:
    descriptor = _dense_descriptor()

    serial_config = _config(
        pe=1,
        simd=1,
    )
    parallel_config = _config(
        pe=4,
        simd=4,
    )

    serial_resources = (
        estimate_resources_from_descriptors(
            [descriptor],
            serial_config,
        )
    )
    parallel_resources = (
        estimate_resources_from_descriptors(
            [descriptor],
            parallel_config,
        )
    )

    serial_performance = estimate_performance(
        resource_estimate=serial_resources,
        raw_cfg=serial_config,
    )
    parallel_performance = estimate_performance(
        resource_estimate=parallel_resources,
        raw_cfg=parallel_config,
    )

    assert parallel_performance[
        "predicted_compute_cycles"
    ] < serial_performance[
        "predicted_compute_cycles"
    ]


def test_performance_cycle_scale_is_applied() -> None:
    config = _config()

    resources = (
        estimate_resources_from_descriptors(
            [_dense_descriptor()],
            config,
        )
    )

    baseline = estimate_performance(
        resource_estimate=resources,
        raw_cfg=config,
    )

    config["analysis"]["design_space"][
        "calibration"
    ]["performance"]["cycle_scale"] = 2.0

    scaled = estimate_performance(
        resource_estimate=resources,
        raw_cfg=config,
    )

    assert scaled["predicted_cycles"] == (
        baseline["predicted_cycles"]
        * 2.0
    )
    assert scaled["predicted_latency_ms"] == (
        baseline["predicted_latency_ms"]
        * 2.0
    )


def test_parse_hls_xml_report(
    tmp_path: Path,
) -> None:
    report = (
        tmp_path / "deeplearn_csynth.xml"
    )

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
      <Average-caseLatency>
        14800
      </Average-caseLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
</Report>
""",
        encoding="utf-8",
    )

    parsed = parse_hls_csynth_report(
        report
    )

    assert parsed["actual_lut"] == 18445
    assert parsed["actual_ff"] == 19822
    assert parsed["actual_dsp"] == 165
    assert parsed["actual_bram18"] == 30
    assert parsed[
        "actual_latency_cycles"
    ] == 14800


def test_estimate_comparison_writes_calibration(
    tmp_path: Path,
) -> None:
    report = (
        tmp_path / "deeplearn_csynth.xml"
    )

    report.write_text(
        """
<Report>
  <LUT>18000</LUT>
  <FF>20000</FF>
  <DSP>160</DSP>
  <BRAM_18K>32</BRAM_18K>
  <Average-caseLatency>
    15000
  </Average-caseLatency>
</Report>
""",
        encoding="utf-8",
    )

    estimate = {
        "predicted_lut": 15000,
        "predicted_ff": 16000,
        "predicted_dsp": 128,
        "predicted_bram18": 30,
        "predicted_cycles": 12000.0,
        "predicted_latency_ms": 0.06,
    }

    result = run_estimate_vs_hls_compare(
        out_dir=tmp_path,
        design_space_summary=estimate,
        csynth_report_path=report,
        clock_mhz=200.0,
    )

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8"
        )
    )

    recommendation = payload[
        "calibration_recommendation"
    ]

    assert payload["available"] is True

    assert recommendation["resources"][
        "lut_scale"
    ] == 1.2

    assert recommendation["resources"][
        "ff_scale"
    ] == 1.25

    assert recommendation["resources"][
        "dsp_scale"
    ] == 1.25

    assert recommendation["performance"][
        "cycle_scale"
    ] == 1.25

    assert "Suggested calibration scales" in (
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
        csynth_report_path=(
            tmp_path / "missing.xml"
        ),
        clock_mhz=200.0,
    )

    payload = json.loads(
        result.results_json.read_text(
            encoding="utf-8"
        )
    )

    assert payload["available"] is False
    assert payload["actual"] is None