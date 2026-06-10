from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple


FLOAT_BYTES = 4


def _cfg_get(
    data: Mapping[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]

    return current


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result <= 0.0:
        return default

    return result


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result < 0.0:
        return default

    return result


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default

    return max(1, result)


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0

    denominator = max(1, denominator)
    return (numerator + denominator - 1) // denominator


def _output_elements(layer: Mapping[str, Any]) -> int:
    dimensions = layer.get("dimensions", {})

    if isinstance(dimensions, Mapping):
        value = int(dimensions.get("output_elements", 0) or 0)

        if value > 0:
            return value

        output_features = int(
            dimensions.get("output_features", 0) or 0
        )

        if output_features > 0:
            return output_features

        output_channels = int(
            dimensions.get("output_channels", 0) or 0
        )
        output_height = int(
            dimensions.get("output_height", 0) or 0
        )
        output_width = int(
            dimensions.get("output_width", 0) or 0
        )

        if output_channels and output_height and output_width:
            return (
                output_channels
                * output_height
                * output_width
            )

    return _ceil_div(
        int(layer.get("activation_bytes_out", 0) or 0),
        FLOAT_BYTES,
    )


def _input_elements(layer: Mapping[str, Any]) -> int:
    dimensions = layer.get("dimensions", {})

    if isinstance(dimensions, Mapping):
        value = int(dimensions.get("input_elements", 0) or 0)

        if value > 0:
            return value

        input_features = int(
            dimensions.get("input_features", 0) or 0
        )

        if input_features > 0:
            return input_features

        input_channels = int(
            dimensions.get("input_channels", 0) or 0
        )
        input_height = int(
            dimensions.get("input_height", 0) or 0
        )
        input_width = int(
            dimensions.get("input_width", 0) or 0
        )

        if input_channels and input_height and input_width:
            return (
                input_channels
                * input_height
                * input_width
            )

    return _ceil_div(
        int(layer.get("activation_bytes_in", 0) or 0),
        FLOAT_BYTES,
    )


def _pipeline_ii(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> int:
    op_type = str(layer.get("op_type", ""))

    configured = _cfg_get(
        raw_cfg,
        f"hls.{op_type.lower()}.pipeline_ii",
        None,
    )

    if configured is None:
        configured = _cfg_get(
            raw_cfg,
            "optimization.parallel.pipeline_ii",
            1,
        )

    return _positive_int(configured, 1)


def _dense_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    dimensions = layer.get("dimensions", {})

    input_features = int(
        dimensions.get("input_features", 0) or 0
    )
    output_features = int(
        dimensions.get("output_features", 0) or 0
    )
    pe = _positive_int(layer.get("pe", 1))
    simd = _positive_int(layer.get("simd", 1))
    ii = _pipeline_ii(layer, raw_cfg)

    output_blocks = _ceil_div(output_features, pe)
    input_blocks = _ceil_div(input_features, simd)

    initialization_cycles = output_blocks
    accumulation_cycles = (
        output_blocks
        * input_blocks
        * ii
    )
    write_cycles = output_blocks

    multiplier_latency = max(
        1,
        _positive_int(
            _cfg_get(
                raw_cfg,
                "analysis.design_space.estimator."
                "dense_multiplier_latency",
                3,
            ),
            3,
        ),
    )
    accumulator_latency = max(
        1,
        _positive_int(
            _cfg_get(
                raw_cfg,
                "analysis.design_space.estimator."
                "dense_accumulator_latency",
                2,
            ),
            2,
        ),
    )
    pipeline_fill_cycles = (
        multiplier_latency
        + accumulator_latency
        + simd
    )

    total = (
        initialization_cycles
        + accumulation_cycles
        + write_cycles
        + pipeline_fill_cycles
    )

    return {
        "initialization_cycles": float(initialization_cycles),
        "accumulation_cycles": float(accumulation_cycles),
        "write_cycles": float(write_cycles),
        "pipeline_fill_cycles": float(pipeline_fill_cycles),
        "control_cycles": 0.0,
        "predicted_cycles": float(total),
    }


def _conv_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    dimensions = layer.get("dimensions", {})

    input_channels = int(
        dimensions.get("input_channels", 0) or 0
    )
    output_channels = int(
        dimensions.get("output_channels", 0) or 0
    )
    output_height = int(
        dimensions.get("output_height", 0) or 0
    )
    output_width = int(
        dimensions.get("output_width", 0) or 0
    )
    kernel_height = int(
        dimensions.get("kernel_height", 0) or 0
    )
    kernel_width = int(
        dimensions.get("kernel_width", 0) or 0
    )
    groups = max(
        1,
        int(dimensions.get("groups", 1) or 1),
    )

    pe = _positive_int(layer.get("pe", 1))
    simd = _positive_int(layer.get("simd", 1))
    ii = _pipeline_ii(layer, raw_cfg)

    channels_per_group = _ceil_div(
        input_channels,
        groups,
    )
    output_channel_blocks = _ceil_div(
        output_channels,
        pe,
    )
    input_channel_blocks = _ceil_div(
        channels_per_group,
        simd,
    )

    output_pixels = output_height * output_width
    kernel_elements = kernel_height * kernel_width

    initialization_cycles = (
        output_channel_blocks
        * output_pixels
    )
    accumulation_cycles = (
        output_channel_blocks
        * output_pixels
        * kernel_elements
        * input_channel_blocks
        * ii
    )
    write_cycles = (
        output_channel_blocks
        * output_pixels
    )

    line_buffer_fill_cycles = (
        max(0, kernel_height - 1)
        * max(0, int(dimensions.get("input_width", 0) or 0))
        * max(1, input_channels)
    )

    multiplier_latency = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "conv_multiplier_latency",
            3,
        ),
        3,
    )
    accumulator_latency = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "conv_accumulator_latency",
            2,
        ),
        2,
    )
    address_latency = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "conv_address_latency",
            3,
        ),
        3,
    )

    pipeline_fill_cycles = (
        multiplier_latency
        + accumulator_latency
        + address_latency
        + simd
        + pe
    )

    total = (
        line_buffer_fill_cycles
        + initialization_cycles
        + accumulation_cycles
        + write_cycles
        + pipeline_fill_cycles
    )

    return {
        "line_buffer_fill_cycles": float(
            line_buffer_fill_cycles
        ),
        "initialization_cycles": float(initialization_cycles),
        "accumulation_cycles": float(accumulation_cycles),
        "write_cycles": float(write_cycles),
        "pipeline_fill_cycles": float(pipeline_fill_cycles),
        "control_cycles": 0.0,
        "predicted_cycles": float(total),
    }


def _pool_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    dimensions = layer.get("dimensions", {})

    output_elements = _output_elements(layer)
    kernel_elements = int(
        dimensions.get("kernel_elements", 0) or 0
    )

    if kernel_elements <= 0:
        attrs = layer.get("attrs", {})
        kernel_shape = (
            attrs.get("kernel_shape", [2, 2])
            if isinstance(attrs, Mapping)
            else [2, 2]
        )

        try:
            kernel_elements = (
                int(kernel_shape[0])
                * int(kernel_shape[1])
            )
        except (TypeError, ValueError, IndexError):
            kernel_elements = 4

    lanes = _positive_int(
        layer.get("multiplier_lanes", 1)
    )
    ii = _pipeline_ii(layer, raw_cfg)

    output_groups = _ceil_div(output_elements, lanes)
    window_cycles = (
        output_groups
        * max(1, kernel_elements)
        * ii
    )

    kernel_height = int(
        dimensions.get("kernel_height", 1) or 1
    )
    input_width = int(
        dimensions.get("input_width", 0) or 0
    )
    input_channels = int(
        dimensions.get("input_channels", 1) or 1
    )

    line_buffer_fill_cycles = (
        max(0, kernel_height - 1)
        * input_width
        * input_channels
    )
    pipeline_fill_cycles = (
        max(1, int(math.ceil(math.log2(max(2, kernel_elements)))))
        + lanes
    )

    total = (
        line_buffer_fill_cycles
        + window_cycles
        + pipeline_fill_cycles
    )

    return {
        "line_buffer_fill_cycles": float(
            line_buffer_fill_cycles
        ),
        "window_cycles": float(window_cycles),
        "pipeline_fill_cycles": float(pipeline_fill_cycles),
        "control_cycles": 0.0,
        "predicted_cycles": float(total),
    }


def _elementwise_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    op_type = str(layer.get("op_type", "Unknown"))
    output_elements = _output_elements(layer)
    lanes = _positive_int(
        layer.get("multiplier_lanes", 1)
    )
    ii = _pipeline_ii(layer, raw_cfg)

    element_groups = _ceil_div(output_elements, lanes)

    operation_latency = {
        "Relu": 1,
        "Add": 1,
        "Flatten": 0,
        "Reshape": 0,
        "LeakyRelu": 3,
        "BatchNormalization": 5,
        "Sigmoid": 12,
    }.get(op_type, 1)

    execution_cycles = element_groups * ii
    pipeline_fill_cycles = operation_latency

    return {
        "execution_cycles": float(execution_cycles),
        "pipeline_fill_cycles": float(pipeline_fill_cycles),
        "control_cycles": 0.0,
        "predicted_cycles": float(
            execution_cycles + pipeline_fill_cycles
        ),
    }


def _softmax_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    output_elements = _output_elements(layer)
    lanes = _positive_int(
        layer.get("multiplier_lanes", 1)
    )
    ii = _pipeline_ii(layer, raw_cfg)

    groups = _ceil_div(output_elements, lanes)

    max_search_cycles = groups * ii
    exponent_cycles = groups * ii
    reduction_cycles = groups * ii
    normalization_cycles = groups * ii

    exp_pipeline_latency = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "softmax_exp_latency",
            12,
        ),
        12,
    )
    division_pipeline_latency = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "softmax_div_latency",
            14,
        ),
        14,
    )

    pipeline_fill_cycles = (
        exp_pipeline_latency
        + division_pipeline_latency
    )

    total = (
        max_search_cycles
        + exponent_cycles
        + reduction_cycles
        + normalization_cycles
        + pipeline_fill_cycles
    )

    return {
        "max_search_cycles": float(max_search_cycles),
        "exponent_cycles": float(exponent_cycles),
        "reduction_cycles": float(reduction_cycles),
        "normalization_cycles": float(
            normalization_cycles
        ),
        "pipeline_fill_cycles": float(
            pipeline_fill_cycles
        ),
        "control_cycles": 0.0,
        "predicted_cycles": float(total),
    }


def _unknown_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    macs = int(layer.get("macs", 0) or 0)
    lanes = _positive_int(
        layer.get("multiplier_lanes", 1)
    )
    ii = _pipeline_ii(layer, raw_cfg)

    if macs > 0:
        execution_cycles = _ceil_div(macs, lanes) * ii
    else:
        execution_cycles = (
            _ceil_div(_output_elements(layer), lanes) * ii
        )

    return {
        "execution_cycles": float(execution_cycles),
        "pipeline_fill_cycles": 1.0,
        "control_cycles": 0.0,
        "predicted_cycles": float(execution_cycles + 1),
    }


def _layer_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    op_type = str(layer.get("op_type", "Unknown"))

    if op_type == "Dense":
        return _dense_schedule(layer, raw_cfg)

    if op_type == "Conv":
        return _conv_schedule(layer, raw_cfg)

    if op_type in {"MaxPool", "AvgPool"}:
        return _pool_schedule(layer, raw_cfg)

    if op_type == "Softmax":
        return _softmax_schedule(layer, raw_cfg)

    if op_type in {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "BatchNormalization",
        "Add",
        "Flatten",
        "Reshape",
    }:
        return _elementwise_schedule(layer, raw_cfg)

    return _unknown_schedule(layer, raw_cfg)


def _input_output_words(
    resource_estimate: Mapping[str, Any],
) -> Tuple[int, int]:
    layers = resource_estimate.get("layers", [])

    if not isinstance(layers, list) or not layers:
        return 0, 0

    first_layer = layers[0]
    last_layer = layers[-1]

    input_words = _ceil_div(
        int(first_layer.get("activation_bytes_in", 0) or 0),
        FLOAT_BYTES,
    )
    output_words = _ceil_div(
        int(last_layer.get("activation_bytes_out", 0) or 0),
        FLOAT_BYTES,
    )

    return input_words, output_words


def _interface_configuration(
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    input_words_per_cycle = _positive_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "input_words_per_cycle",
            1.0,
        ),
        1.0,
    )
    output_words_per_cycle = _positive_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "output_words_per_cycle",
            1.0,
        ),
        1.0,
    )
    input_startup_cycles = _nonnegative_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "input_startup_cycles",
            3.0,
        ),
        3.0,
    )
    output_startup_cycles = _nonnegative_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "output_startup_cycles",
            3.0,
        ),
        3.0,
    )

    return {
        "input_words_per_cycle": input_words_per_cycle,
        "output_words_per_cycle": output_words_per_cycle,
        "input_startup_cycles": input_startup_cycles,
        "output_startup_cycles": output_startup_cycles,
    }


def _top_level_control_cycles(
    layer_count: int,
    raw_cfg: Mapping[str, Any],
) -> float:
    entry_cycles = _nonnegative_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "kernel_entry_cycles",
            4.0,
        ),
        4.0,
    )
    call_cycles_per_layer = _nonnegative_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "call_cycles_per_layer",
            2.0,
        ),
        2.0,
    )
    exit_cycles = _nonnegative_float(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator."
            "kernel_exit_cycles",
            2.0,
        ),
        2.0,
    )

    return (
        entry_cycles
        + layer_count * call_cycles_per_layer
        + exit_cycles
    )


def _explicit_calibration(
    raw_cfg: Mapping[str, Any],
) -> Dict[str, float | bool]:
    configured = _cfg_get(
        raw_cfg,
        "analysis.design_space.calibration.performance",
        {},
    )

    if not isinstance(configured, Mapping) or not configured:
        return {
            "enabled": False,
            "fixed_cycles": 0.0,
            "cycle_scale": 1.0,
            "input_words_per_cycle": 1.0,
            "output_words_per_cycle": 1.0,
        }

    explicit_enabled = configured.get("enabled")
    enabled = (
        bool(explicit_enabled)
        if explicit_enabled is not None
        else True
    )

    return {
        "enabled": enabled,
        "fixed_cycles": _nonnegative_float(
            configured.get("fixed_cycles", 0.0)
        ),
        "cycle_scale": _positive_float(
            configured.get("cycle_scale", 1.0),
            1.0,
        ),
        "input_words_per_cycle": _positive_float(
            configured.get("input_words_per_cycle", 1.0),
            1.0,
        ),
        "output_words_per_cycle": _positive_float(
            configured.get("output_words_per_cycle", 1.0),
            1.0,
        ),
    }


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

    layers = resource_estimate.get("layers", [])

    if not isinstance(layers, list):
        layers = []

    layer_cycle_rows: List[Dict[str, Any]] = []
    compute_cycles = 0.0
    pipeline_fill_cycles = 0.0

    for layer in layers:
        schedule = _layer_schedule(layer, raw_cfg)
        cycles = float(schedule["predicted_cycles"])
        compute_cycles += cycles
        pipeline_fill_cycles += float(
            schedule.get("pipeline_fill_cycles", 0.0)
        )

        layer_cycle_rows.append(
            {
                "layer_index": layer.get("layer_index"),
                "layer_name": layer.get("layer_name"),
                "op_type": layer.get("op_type"),
                "input_elements": _input_elements(layer),
                "output_elements": _output_elements(layer),
                "pe": int(layer.get("pe", 1) or 1),
                "simd": int(layer.get("simd", 1) or 1),
                "pipeline_ii": _pipeline_ii(
                    layer,
                    raw_cfg,
                ),
                **schedule,
            }
        )

    input_words, output_words = _input_output_words(
        resource_estimate
    )
    interface = _interface_configuration(raw_cfg)
    calibration = _explicit_calibration(raw_cfg)

    input_words_per_cycle = float(
        calibration["input_words_per_cycle"]
        if calibration["enabled"]
        else interface["input_words_per_cycle"]
    )
    output_words_per_cycle = float(
        calibration["output_words_per_cycle"]
        if calibration["enabled"]
        else interface["output_words_per_cycle"]
    )

    input_transfer_cycles = (
        interface["input_startup_cycles"]
        + input_words / input_words_per_cycle
    )
    output_transfer_cycles = (
        interface["output_startup_cycles"]
        + output_words / output_words_per_cycle
    )
    transfer_cycles = (
        input_transfer_cycles
        + output_transfer_cycles
    )

    control_cycles = _top_level_control_cycles(
        len(layers),
        raw_cfg,
    )

    analytical_cycles = (
        compute_cycles
        + transfer_cycles
        + control_cycles
    )

    fixed_cycles = (
        float(calibration["fixed_cycles"])
        if calibration["enabled"]
        else 0.0
    )
    unscaled_cycles = analytical_cycles + fixed_cycles
    cycle_scale = (
        float(calibration["cycle_scale"])
        if calibration["enabled"]
        else 1.0
    )
    total_cycles = unscaled_cycles * cycle_scale

    latency_ms = total_cycles / (clock_mhz * 1_000.0)
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
        resource_estimate.get("totals", {}).get(
            "total_macs",
            0,
        )
        or 0
    )
    effective_parallel_macs = (
        total_macs / compute_cycles
        if compute_cycles > 0.0
        else 0.0
    )

    return {
        # Retained for compatibility with existing tests/readers.
        "performance_model": "operator_schedule_calibrated_v1",
        "analytical_performance_model": (
            "operator_execution_schedule_v2"
        ),
        "estimation_mode": (
            "analytical_with_explicit_calibration"
            if calibration["enabled"]
            else "analytical"
        ),
        "clock_mhz": float(clock_mhz),
        "predicted_parallel_macs": float(
            effective_parallel_macs
        ),
        "predicted_compute_cycles": float(compute_cycles),
        "predicted_pipeline_fill_cycles": float(
            pipeline_fill_cycles
        ),
        "predicted_input_transfer_cycles": float(
            input_transfer_cycles
        ),
        "predicted_output_transfer_cycles": float(
            output_transfer_cycles
        ),
        "predicted_transfer_cycles": float(
            transfer_cycles
        ),
        "predicted_control_cycles": float(control_cycles),
        "predicted_analytical_cycles": float(
            analytical_cycles
        ),
        "predicted_fixed_cycles": float(fixed_cycles),
        "predicted_unscaled_cycles": float(
            unscaled_cycles
        ),
        "predicted_cycles": float(total_cycles),
        "predicted_latency_ms": float(latency_ms),
        "predicted_throughput_fps": float(
            throughput_fps
        ),
        "predicted_speedup_vs_cpu": float(
            speedup_vs_cpu
        ),
        "input_words": int(input_words),
        "output_words": int(output_words),
        "interface": interface,
        "calibration": calibration,
        "layer_cycles": layer_cycle_rows,
    }