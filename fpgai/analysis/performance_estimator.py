from __future__ import annotations

import math
from typing import Any, Dict, Mapping


DEFAULT_PERFORMANCE_CALIBRATION = {
    # Top-level AXI, control, pipeline fill, and function-call overhead.
    "fixed_cycles": 2_000.0,

    # Applied after layer and transfer cycle estimation.
    "cycle_scale": 1.0,

    # AXI stream transfers one float32 word per cycle.
    "input_words_per_cycle": 1.0,
    "output_words_per_cycle": 1.0,
}


def _cfg_get(
    data: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if (
            not isinstance(current, Mapping)
            or key not in current
        ):
            return default

        current = current[key]

    return current


def _positive_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if (
        not math.isfinite(result)
        or result <= 0.0
    ):
        return default

    return result


def _nonnegative_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if (
        not math.isfinite(result)
        or result < 0.0
    ):
        return default

    return result


def _ceil_div(
    numerator: int,
    denominator: int,
) -> int:
    if numerator <= 0:
        return 0

    return (
        numerator
        + max(1, denominator)
        - 1
    ) // max(1, denominator)


def _performance_calibration(
    raw_cfg: Dict[str, Any],
) -> Dict[str, float]:
    result = dict(
        DEFAULT_PERFORMANCE_CALIBRATION
    )

    configured = _cfg_get(
        raw_cfg,
        "analysis.design_space.calibration.performance",
        {},
    )

    if isinstance(configured, Mapping):
        result["fixed_cycles"] = (
            _nonnegative_float(
                configured.get(
                    "fixed_cycles",
                    result["fixed_cycles"],
                ),
                result["fixed_cycles"],
            )
        )
        result["cycle_scale"] = (
            _positive_float(
                configured.get(
                    "cycle_scale",
                    result["cycle_scale"],
                ),
                result["cycle_scale"],
            )
        )
        result["input_words_per_cycle"] = (
            _positive_float(
                configured.get(
                    "input_words_per_cycle",
                    result[
                        "input_words_per_cycle"
                    ],
                ),
                result[
                    "input_words_per_cycle"
                ],
            )
        )
        result["output_words_per_cycle"] = (
            _positive_float(
                configured.get(
                    "output_words_per_cycle",
                    result[
                        "output_words_per_cycle"
                    ],
                ),
                result[
                    "output_words_per_cycle"
                ],
            )
        )

    return result


def _dense_cycles(
    layer: Dict[str, Any],
) -> float:
    dimensions = layer.get(
        "dimensions",
        {},
    )

    input_features = int(
        dimensions.get(
            "input_features",
            0,
        )
        or 0
    )
    output_features = int(
        dimensions.get(
            "output_features",
            0,
        )
        or 0
    )
    pe = max(
        1,
        int(
            layer.get(
                "pe",
                1,
            )
            or 1
        ),
    )
    simd = max(
        1,
        int(
            layer.get(
                "simd",
                1,
            )
            or 1
        ),
    )

    output_blocks = _ceil_div(
        output_features,
        pe,
    )
    input_blocks = _ceil_div(
        input_features,
        simd,
    )

    # One initialization, input accumulation, and output
    # write phase per output block.
    return float(
        output_blocks
        * (
            input_blocks
            + 2
        )
    )


def _conv_cycles(
    layer: Dict[str, Any],
) -> float:
    dimensions = layer.get(
        "dimensions",
        {},
    )

    input_channels = int(
        dimensions.get(
            "input_channels",
            0,
        )
        or 0
    )
    output_channels = int(
        dimensions.get(
            "output_channels",
            0,
        )
        or 0
    )
    output_height = int(
        dimensions.get(
            "output_height",
            0,
        )
        or 0
    )
    output_width = int(
        dimensions.get(
            "output_width",
            0,
        )
        or 0
    )
    kernel_height = int(
        dimensions.get(
            "kernel_height",
            0,
        )
        or 0
    )
    kernel_width = int(
        dimensions.get(
            "kernel_width",
            0,
        )
        or 0
    )

    pe = max(
        1,
        int(
            layer.get(
                "pe",
                1,
            )
            or 1
        ),
    )
    simd = max(
        1,
        int(
            layer.get(
                "simd",
                1,
            )
            or 1
        ),
    )

    output_channel_blocks = _ceil_div(
        output_channels,
        pe,
    )
    input_channel_blocks = _ceil_div(
        input_channels,
        simd,
    )

    output_pixels = (
        output_height
        * output_width
    )
    kernel_area = (
        kernel_height
        * kernel_width
    )

    accumulation_cycles = (
        input_channel_blocks
        * kernel_area
    )

    # Each output-channel block processes every output
    # pixel, including accumulator initialization and store.
    return float(
        output_channel_blocks
        * output_pixels
        * (
            accumulation_cycles
            + 2
        )
    )


def _pool_cycles(
    layer: Dict[str, Any],
) -> float:
    dimensions = layer.get(
        "dimensions",
        {},
    )

    output_elements = int(
        dimensions.get(
            "output_elements",
            0,
        )
        or 0
    )

    if output_elements <= 0:
        activation_bytes_out = int(
            layer.get(
                "activation_bytes_out",
                0,
            )
            or 0
        )
        output_elements = _ceil_div(
            activation_bytes_out,
            4,
        )

    attrs = layer.get(
        "attrs",
        {},
    )

    kernel_shape = attrs.get(
        "kernel_shape",
        [2, 2],
    )

    try:
        kernel_area = (
            int(kernel_shape[0])
            * int(kernel_shape[1])
        )
    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        kernel_area = 4

    return float(
        output_elements
        * max(1, kernel_area)
    )


def _elementwise_cycles(
    layer: Dict[str, Any],
) -> float:
    output_elements = int(
        layer.get(
            "dimensions",
            {},
        ).get(
            "output_elements",
            0,
        )
        or 0
    )

    if output_elements <= 0:
        output_elements = _ceil_div(
            int(
                layer.get(
                    "activation_bytes_out",
                    0,
                )
                or 0
            ),
            4,
        )

    lanes = max(
        1,
        int(
            layer.get(
                "multiplier_lanes",
                1,
            )
            or 1
        ),
    )

    return float(
        _ceil_div(
            output_elements,
            lanes,
        )
    )


def _softmax_cycles(
    layer: Dict[str, Any],
) -> float:
    output_elements = int(
        layer.get(
            "dimensions",
            {},
        ).get(
            "output_elements",
            0,
        )
        or 0
    )

    if output_elements <= 0:
        output_elements = _ceil_div(
            int(
                layer.get(
                    "activation_bytes_out",
                    0,
                )
                or 0
            ),
            4,
        )

    # Maximum search, exponential approximation,
    # reduction, and normalization.
    return float(
        max(
            1,
            4 * output_elements,
        )
    )


def _layer_cycles(
    layer: Dict[str, Any],
) -> float:
    op_type = str(
        layer.get(
            "op_type",
            "Unknown",
        )
    )

    if op_type == "Dense":
        return _dense_cycles(layer)

    if op_type == "Conv":
        return _conv_cycles(layer)

    if op_type in {
        "MaxPool",
        "AvgPool",
    }:
        return _pool_cycles(layer)

    if op_type == "Softmax":
        return _softmax_cycles(layer)

    if op_type in {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "BatchNormalization",
        "Add",
        "Flatten",
        "Reshape",
    }:
        return _elementwise_cycles(layer)

    macs = int(
        layer.get(
            "macs",
            0,
        )
        or 0
    )
    lanes = max(
        1,
        int(
            layer.get(
                "multiplier_lanes",
                1,
            )
            or 1
        ),
    )

    if macs > 0:
        return float(
            _ceil_div(
                macs,
                lanes,
            )
        )

    return _elementwise_cycles(layer)


def _input_output_words(
    resource_estimate: Dict[str, Any],
) -> tuple[int, int]:
    layers = resource_estimate.get(
        "layers",
        [],
    )

    if not layers:
        return 0, 0

    first_layer = layers[0]
    last_layer = layers[-1]

    input_words = _ceil_div(
        int(
            first_layer.get(
                "activation_bytes_in",
                0,
            )
            or 0
        ),
        4,
    )
    output_words = _ceil_div(
        int(
            last_layer.get(
                "activation_bytes_out",
                0,
            )
            or 0
        ),
        4,
    )

    return input_words, output_words


def estimate_performance(
    *,
    resource_estimate: Dict[str, Any],
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    clock_mhz = _positive_float(
        _cfg_get(
            raw_cfg,
            "targets.platform.clocks.0.target_mhz",
            200.0,
        ),
        200.0,
    )
    cpu_baseline_ms = _positive_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.performance."
            "baseline_cpu_latency_ms",
            1.0,
        ),
        1.0,
    )
    calibration = _performance_calibration(
        raw_cfg
    )

    layers = resource_estimate.get(
        "layers",
        [],
    )

    layer_cycle_rows = []
    raw_compute_cycles = 0.0

    for layer in layers:
        cycles = _layer_cycles(layer)
        raw_compute_cycles += cycles

        layer_cycle_rows.append(
            {
                "layer_index": layer.get(
                    "layer_index"
                ),
                "layer_name": layer.get(
                    "layer_name"
                ),
                "op_type": layer.get(
                    "op_type"
                ),
                "predicted_cycles": cycles,
            }
        )

    input_words, output_words = (
        _input_output_words(
            resource_estimate
        )
    )

    input_transfer_cycles = (
        input_words
        / calibration[
            "input_words_per_cycle"
        ]
    )
    output_transfer_cycles = (
        output_words
        / calibration[
            "output_words_per_cycle"
        ]
    )

    transfer_cycles = (
        input_transfer_cycles
        + output_transfer_cycles
    )

    unscaled_cycles = (
        calibration["fixed_cycles"]
        + raw_compute_cycles
        + transfer_cycles
    )
    total_cycles = (
        unscaled_cycles
        * calibration["cycle_scale"]
    )

    latency_ms = (
        total_cycles
        / (
            clock_mhz
            * 1_000.0
        )
    )

    throughput_fps = (
        1_000.0 / latency_ms
        if latency_ms > 0.0
        else 0.0
    )
    speedup_vs_cpu = (
        cpu_baseline_ms / latency_ms
        if latency_ms > 0.0
        else 0.0
    )

    total_macs = int(
        resource_estimate.get(
            "totals",
            {},
        ).get(
            "total_macs",
            0,
        )
        or 0
    )

    effective_parallel_macs = (
        total_macs / raw_compute_cycles
        if raw_compute_cycles > 0.0
        else 0.0
    )

    return {
        "performance_model": (
            "operator_schedule_calibrated_v1"
        ),
        "clock_mhz": float(
            clock_mhz
        ),
        "predicted_parallel_macs": float(
            effective_parallel_macs
        ),
        "predicted_compute_cycles": float(
            raw_compute_cycles
        ),
        "predicted_transfer_cycles": float(
            transfer_cycles
        ),
        "predicted_fixed_cycles": float(
            calibration["fixed_cycles"]
        ),
        "predicted_unscaled_cycles": float(
            unscaled_cycles
        ),
        "predicted_cycles": float(
            total_cycles
        ),
        "predicted_latency_ms": float(
            latency_ms
        ),
        "predicted_throughput_fps": float(
            throughput_fps
        ),
        "predicted_speedup_vs_cpu": float(
            speedup_vs_cpu
        ),
        "input_words": int(
            input_words
        ),
        "output_words": int(
            output_words
        ),
        "calibration": calibration,
        "layer_cycles": layer_cycle_rows,
    }