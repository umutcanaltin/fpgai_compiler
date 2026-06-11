from __future__ import annotations

from copy import deepcopy

from fpgai.analysis.architecture_schedule_model import (
    estimate_architecture_layer_schedule,
    estimate_architecture_schedules,
)
from fpgai.analysis.performance_estimator import (
    estimate_performance,
)


def _dense_layer() -> dict:
    return {
        "layer_index": 0,
        "layer_name": "dense0",
        "op_type": "Dense",
        "pe": 2,
        "simd": 4,
        "multiplier_lanes": 8,
        "activation_bytes_in": 128 * 4,
        "activation_bytes_out": 10 * 4,
        "dimensions": {
            "input_features": 128,
            "output_features": 10,
            "input_elements": 128,
            "output_elements": 10,
            "macs": 1280,
        },
        "architecture": {
            "dimensions": {
                "input_features": 128,
                "output_features": 10,
                "input_elements": 128,
                "output_elements": 10,
                "macs": 1280,
            },
            "pipeline_scope": "input_base",
            "pipeline_ii": 1,
            "pipeline_overlap": 1,
            "reduction_iterations": 32,
            "explicit_lanes": 8,
            "effective_lanes": 8,
            "unroll": {
                "out": 2,
                "in": 4,
            },
        },
    }


def _conv_layer() -> dict:
    return {
        "layer_index": 0,
        "layer_name": "conv0",
        "op_type": "Conv",
        "pe": 2,
        "simd": 2,
        "multiplier_lanes": 72,
        "activation_bytes_in": 4 * 28 * 28 * 4,
        "activation_bytes_out": 8 * 26 * 26 * 4,
        "dimensions": {
            "input_channels": 4,
            "input_height": 28,
            "input_width": 28,
            "output_channels": 8,
            "output_height": 26,
            "output_width": 26,
            "channels_per_group": 4,
            "kernel_elements": 9,
            "input_elements": 4 * 28 * 28,
            "output_elements": 8 * 26 * 26,
            "macs": 8 * 26 * 26 * 4 * 9,
        },
        "architecture": {
            "dimensions": {
                "input_channels": 4,
                "input_height": 28,
                "input_width": 28,
                "output_channels": 8,
                "output_height": 26,
                "output_width": 26,
                "channels_per_group": 4,
                "kernel_elements": 9,
                "input_elements": 4 * 28 * 28,
                "output_elements": 8 * 26 * 26,
                "macs": 8 * 26 * 26 * 4 * 9,
            },
            "pipeline_scope": "output_column",
            "pipeline_ii": 1,
            "pipeline_overlap": 18,
            "reduction_iterations": 18,
            "explicit_lanes": 4,
            "effective_lanes": 72,
            "unroll": {
                "oc": 2,
                "ic": 2,
            },
        },
    }


def _relu_layer() -> dict:
    return {
        "layer_index": 1,
        "layer_name": "relu0",
        "op_type": "Relu",
        "pe": 1,
        "simd": 1,
        "multiplier_lanes": 0,
        "activation_bytes_in": 10 * 4,
        "activation_bytes_out": 10 * 4,
        "dimensions": {
            "input_elements": 10,
            "output_elements": 10,
            "macs": 0,
        },
        "architecture": {
            "dimensions": {
                "input_elements": 10,
                "output_elements": 10,
                "macs": 0,
            },
            "pipeline_scope": "element",
            "pipeline_ii": 1,
            "pipeline_overlap": 1,
            "reduction_iterations": 1,
            "explicit_lanes": 1,
            "effective_lanes": 1,
            "unroll": {
                "element": 1,
            },
        },
    }


def _resource_estimate(
    layers: list[dict],
    *,
    execution_mode: str,
) -> dict:
    return {
        "architecture": {
            "policy": "Balanced",
            "execution_mode": execution_mode,
            "clock_mhz": 200.0,
        },
        "totals": {
            "total_macs": sum(
                int(layer["dimensions"]["macs"])
                for layer in layers
            ),
        },
        "layers": layers,
    }


def _config() -> dict:
    return {
        "targets": {
            "platform": {
                "clocks": [
                    {
                        "name": "pl_clk0",
                        "target_mhz": 200.0,
                    }
                ]
            }
        },
        "analysis": {
            "design_space": {
                "performance": {
                    "baseline_cpu_latency_ms": 1.0,
                },
                "estimator": {
                    "input_words_per_cycle": 1.0,
                    "output_words_per_cycle": 1.0,
                    "input_startup_cycles": 3.0,
                    "output_startup_cycles": 3.0,
                    "kernel_entry_cycles": 4.0,
                    "call_cycles_per_layer": 2.0,
                    "kernel_exit_cycles": 2.0,
                },
            }
        },
    }


def test_dense_schedule_uses_planned_unroll_and_ii() -> None:
    schedule = estimate_architecture_layer_schedule(
        _dense_layer(),
        _config(),
    )

    assert schedule["pipeline_scope"] == "input_base"
    assert schedule["pipeline_ii"] == 1
    assert schedule["input_blocks"] == 32
    assert schedule["output_blocks"] == 5

    assert schedule["initialization_cycles"] == 5
    assert schedule["accumulation_cycles"] == 160
    assert schedule["write_cycles"] == 5
    assert schedule["pipeline_fill_cycles"] == 9
    assert schedule["predicted_cycles"] == 179


def test_dense_higher_ii_increases_cycles() -> None:
    fast_layer = _dense_layer()
    slow_layer = deepcopy(fast_layer)
    slow_layer["architecture"]["pipeline_ii"] = 2

    fast = estimate_architecture_layer_schedule(
        fast_layer,
        _config(),
    )
    slow = estimate_architecture_layer_schedule(
        slow_layer,
        _config(),
    )

    assert slow["accumulation_cycles"] == (
        fast["accumulation_cycles"] * 2
    )
    assert (
        slow["predicted_cycles"]
        > fast["predicted_cycles"]
    )


def test_conv_schedule_models_output_column_pipeline() -> None:
    schedule = estimate_architecture_layer_schedule(
        _conv_layer(),
        _config(),
    )

    assert schedule["pipeline_scope"] == "output_column"
    assert schedule["output_channel_blocks"] == 4
    assert schedule["input_channel_blocks"] == 2
    assert schedule["output_positions"] == 26 * 26
    assert schedule["reduction_iterations"] == 18

    assert schedule["accumulation_cycles"] == (
        4 * 26 * 26
    )
    assert schedule["pipeline_fill_cycles"] == 25
    assert schedule["predicted_cycles"] == (
        4 + 4 * 26 * 26 + 4 + 25
    )


def test_schedule_collection_preserves_layer_identity() -> None:
    schedules = estimate_architecture_schedules(
        [
            _dense_layer(),
            _relu_layer(),
        ],
        _config(),
    )

    assert len(schedules) == 2
    assert schedules[0]["layer_index"] == 0
    assert schedules[0]["layer_name"] == "dense0"
    assert schedules[0]["op_type"] == "Dense"
    assert schedules[1]["layer_index"] == 1
    assert schedules[1]["layer_name"] == "relu0"
    assert schedules[1]["op_type"] == "Relu"


def test_sequential_performance_sums_layer_cycles() -> None:
    layers = [
        _dense_layer(),
        _relu_layer(),
    ]

    result = estimate_performance(
        resource_estimate=_resource_estimate(
            layers,
            execution_mode="sequential",
        ),
        raw_cfg=_config(),
    )

    layer_total = sum(
        float(row["predicted_cycles"])
        for row in result["layer_cycles"]
    )

    assert result["execution_mode"] == "sequential"
    assert result["predicted_compute_cycles"] == layer_total
    assert (
        result["predicted_sequential_compute_cycles"]
        == layer_total
    )
    assert result["predicted_transfer_cycles"] == (
        result["predicted_input_transfer_cycles"]
        + result["predicted_output_transfer_cycles"]
    )


def test_dataflow_performance_overlaps_stages_and_transfers() -> None:
    layers = [
        _dense_layer(),
        _relu_layer(),
    ]

    sequential = estimate_performance(
        resource_estimate=_resource_estimate(
            layers,
            execution_mode="sequential",
        ),
        raw_cfg=_config(),
    )
    dataflow = estimate_performance(
        resource_estimate=_resource_estimate(
            layers,
            execution_mode="dataflow",
        ),
        raw_cfg=_config(),
    )

    assert dataflow["execution_mode"] == "dataflow"

    assert (
        dataflow["predicted_compute_cycles"]
        < dataflow[
            "predicted_sequential_compute_cycles"
        ]
    )
    assert (
        dataflow["predicted_transfer_cycles"]
        == max(
            dataflow[
                "predicted_input_transfer_cycles"
            ],
            dataflow[
                "predicted_output_transfer_cycles"
            ],
        )
    )
    assert (
        dataflow["predicted_cycles"]
        < sequential["predicted_cycles"]
    )


def test_performance_uses_architecture_clock() -> None:
    resource_estimate = _resource_estimate(
        [_dense_layer()],
        execution_mode="sequential",
    )
    resource_estimate["architecture"][
        "clock_mhz"
    ] = 250.0

    result = estimate_performance(
        resource_estimate=resource_estimate,
        raw_cfg=_config(),
    )

    assert result["clock_mhz"] == 250.0
    assert result["predicted_latency_ms"] == (
        result["predicted_cycles"]
        / 250_000.0
    )