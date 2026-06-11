from __future__ import annotations

from typing import Any, Mapping, Sequence


def _get(
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


def _positive_int(
    value: Any,
    default: int = 1,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return max(1, default)

    return max(1, result)


def _nonnegative_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return max(0, default)

    return max(0, result)


def _ceil_div(
    value: int,
    divisor: int,
) -> int:
    if value <= 0:
        return 0

    divisor = max(1, divisor)
    return (value + divisor - 1) // divisor


def _architecture(
    layer: Mapping[str, Any],
) -> Mapping[str, Any]:
    architecture = layer.get("architecture", {})

    if isinstance(architecture, Mapping):
        return architecture

    return {}


def _dimensions(
    layer: Mapping[str, Any],
) -> Mapping[str, Any]:
    architecture = _architecture(layer)
    dimensions = architecture.get(
        "dimensions",
        layer.get("dimensions", {}),
    )

    if isinstance(dimensions, Mapping):
        return dimensions

    return {}


def _unroll(
    layer: Mapping[str, Any],
) -> Mapping[str, Any]:
    architecture = _architecture(layer)
    unroll = architecture.get("unroll", {})

    if isinstance(unroll, Mapping):
        return unroll

    return {}


def _pipeline_ii(
    layer: Mapping[str, Any],
) -> int:
    architecture = _architecture(layer)

    return _positive_int(
        architecture.get(
            "pipeline_ii",
            layer.get("pipeline_ii", 1),
        )
    )


def _pipeline_scope(
    layer: Mapping[str, Any],
) -> str:
    architecture = _architecture(layer)

    return str(
        architecture.get(
            "pipeline_scope",
            "unknown",
        )
    )


def _pipeline_overlap(
    layer: Mapping[str, Any],
) -> int:
    architecture = _architecture(layer)

    return _positive_int(
        architecture.get(
            "pipeline_overlap",
            1,
        )
    )


def _reduction_iterations(
    layer: Mapping[str, Any],
) -> int:
    architecture = _architecture(layer)

    return _positive_int(
        architecture.get(
            "reduction_iterations",
            1,
        )
    )


def _explicit_lanes(
    layer: Mapping[str, Any],
) -> int:
    architecture = _architecture(layer)

    return _positive_int(
        architecture.get(
            "explicit_lanes",
            layer.get("multiplier_lanes", 1),
        )
    )


def _input_elements(
    layer: Mapping[str, Any],
) -> int:
    dimensions = _dimensions(layer)
    value = _nonnegative_int(
        dimensions.get("input_elements", 0)
    )

    if value > 0:
        return value

    return _ceil_div(
        _nonnegative_int(
            layer.get("activation_bytes_in", 0)
        ),
        4,
    )


def _output_elements(
    layer: Mapping[str, Any],
) -> int:
    dimensions = _dimensions(layer)
    value = _nonnegative_int(
        dimensions.get("output_elements", 0)
    )

    if value > 0:
        return value

    return _ceil_div(
        _nonnegative_int(
            layer.get("activation_bytes_out", 0)
        ),
        4,
    )


def _latency_setting(
    raw_cfg: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    return _positive_int(
        _get(
            raw_cfg,
            (
                "analysis.design_space.estimator."
                f"{name}"
            ),
            default,
        ),
        default,
    )


def _base_result(
    layer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_scope": _pipeline_scope(layer),
        "pipeline_ii": _pipeline_ii(layer),
        "pipeline_overlap": _pipeline_overlap(layer),
        "reduction_iterations": _reduction_iterations(layer),
        "explicit_lanes": _explicit_lanes(layer),
        "initialization_cycles": 0.0,
        "accumulation_cycles": 0.0,
        "write_cycles": 0.0,
        "pipeline_fill_cycles": 0.0,
        "control_cycles": 0.0,
        "predicted_cycles": 0.0,
    }


def _dense_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result(layer)
    dimensions = _dimensions(layer)
    unroll = _unroll(layer)

    input_features = _nonnegative_int(
        dimensions.get("input_features", 0)
    )
    output_features = _nonnegative_int(
        dimensions.get("output_features", 0)
    )
    input_unroll = _positive_int(
        unroll.get(
            "in",
            layer.get("simd", 1),
        )
    )
    output_unroll = _positive_int(
        unroll.get(
            "out",
            layer.get("pe", 1),
        )
    )
    pipeline_ii = _pipeline_ii(layer)

    input_blocks = _ceil_div(
        input_features,
        input_unroll,
    )
    output_blocks = _ceil_div(
        output_features,
        output_unroll,
    )

    initialization_cycles = output_blocks
    accumulation_cycles = (
        output_blocks
        * input_blocks
        * pipeline_ii
    )
    write_cycles = output_blocks

    multiplier_latency = _latency_setting(
        raw_cfg,
        "dense_multiplier_latency",
        3,
    )
    accumulator_latency = _latency_setting(
        raw_cfg,
        "dense_accumulator_latency",
        2,
    )
    pipeline_fill_cycles = (
        multiplier_latency
        + accumulator_latency
        + input_unroll
    )

    total = (
        initialization_cycles
        + accumulation_cycles
        + write_cycles
        + pipeline_fill_cycles
    )

    result.update(
        {
            "input_blocks": input_blocks,
            "output_blocks": output_blocks,
            "initialization_cycles": float(
                initialization_cycles
            ),
            "accumulation_cycles": float(
                accumulation_cycles
            ),
            "write_cycles": float(
                write_cycles
            ),
            "pipeline_fill_cycles": float(
                pipeline_fill_cycles
            ),
            "predicted_cycles": float(total),
        }
    )

    return result


def _conv_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result(layer)
    dimensions = _dimensions(layer)
    unroll = _unroll(layer)

    output_channels = _nonnegative_int(
        dimensions.get("output_channels", 0)
    )
    output_height = _nonnegative_int(
        dimensions.get("output_height", 0)
    )
    output_width = _nonnegative_int(
        dimensions.get("output_width", 0)
    )
    channels_per_group = _nonnegative_int(
        dimensions.get(
            "channels_per_group",
            dimensions.get("input_channels", 0),
        )
    )
    kernel_elements = _positive_int(
        dimensions.get("kernel_elements", 1)
    )

    output_unroll = _positive_int(
        unroll.get(
            "oc",
            layer.get("pe", 1),
        )
    )
    input_unroll = _positive_int(
        unroll.get(
            "ic",
            layer.get("simd", 1),
        )
    )
    pipeline_ii = _pipeline_ii(layer)

    output_channel_blocks = _ceil_div(
        output_channels,
        output_unroll,
    )
    input_channel_blocks = _ceil_div(
        channels_per_group,
        input_unroll,
    )
    output_positions = (
        output_height
        * output_width
    )
    reduction_iterations = (
        input_channel_blocks
        * kernel_elements
    )

    # Conv pipelines the output-column loop. The complete reduction
    # body is therefore pipeline depth, not repeated serial latency.
    accumulation_cycles = (
        output_channel_blocks
        * output_positions
        * pipeline_ii
    )

    multiplier_latency = _latency_setting(
        raw_cfg,
        "conv_multiplier_latency",
        3,
    )
    accumulator_latency = _latency_setting(
        raw_cfg,
        "conv_accumulator_latency",
        2,
    )
    address_latency = _latency_setting(
        raw_cfg,
        "conv_address_latency",
        2,
    )

    pipeline_fill_cycles = (
        reduction_iterations
        + multiplier_latency
        + accumulator_latency
        + address_latency
    )
    initialization_cycles = output_channel_blocks
    write_cycles = output_channel_blocks

    total = (
        initialization_cycles
        + accumulation_cycles
        + write_cycles
        + pipeline_fill_cycles
    )

    result.update(
        {
            "output_channel_blocks": output_channel_blocks,
            "input_channel_blocks": input_channel_blocks,
            "output_positions": output_positions,
            "reduction_iterations": reduction_iterations,
            "initialization_cycles": float(
                initialization_cycles
            ),
            "accumulation_cycles": float(
                accumulation_cycles
            ),
            "write_cycles": float(
                write_cycles
            ),
            "pipeline_fill_cycles": float(
                pipeline_fill_cycles
            ),
            "predicted_cycles": float(total),
        }
    )

    return result


def _pool_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result(layer)
    dimensions = _dimensions(layer)

    output_elements = _output_elements(layer)
    kernel_elements = _positive_int(
        dimensions.get("kernel_elements", 1)
    )
    pipeline_ii = _pipeline_ii(layer)

    accumulation_cycles = (
        output_elements
        * pipeline_ii
    )
    pipeline_fill_cycles = (
        kernel_elements
        + _latency_setting(
            raw_cfg,
            "pool_compare_latency",
            2,
        )
    )

    if str(layer.get("op_type", "")) == "AvgPool":
        pipeline_fill_cycles += _latency_setting(
            raw_cfg,
            "pool_divider_latency",
            4,
        )

    total = (
        accumulation_cycles
        + pipeline_fill_cycles
    )

    result.update(
        {
            "output_positions": output_elements,
            "kernel_elements": kernel_elements,
            "accumulation_cycles": float(
                accumulation_cycles
            ),
            "pipeline_fill_cycles": float(
                pipeline_fill_cycles
            ),
            "predicted_cycles": float(total),
        }
    )

    return result


def _softmax_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result(layer)

    elements = _output_elements(layer)
    pipeline_ii = _pipeline_ii(layer)

    max_cycles = elements * pipeline_ii
    exponential_cycles = elements * pipeline_ii
    normalization_cycles = elements * pipeline_ii

    exponential_latency = _latency_setting(
        raw_cfg,
        "softmax_exponential_latency",
        12,
    )
    divider_latency = _latency_setting(
        raw_cfg,
        "softmax_divider_latency",
        12,
    )
    reduction_latency = _latency_setting(
        raw_cfg,
        "softmax_reduction_latency",
        4,
    )

    pipeline_fill_cycles = (
        exponential_latency
        + divider_latency
        + reduction_latency
    )
    accumulation_cycles = (
        max_cycles
        + exponential_cycles
        + normalization_cycles
    )
    total = (
        accumulation_cycles
        + pipeline_fill_cycles
    )

    result.update(
        {
            "passes": 3,
            "max_cycles": float(max_cycles),
            "exponential_cycles": float(
                exponential_cycles
            ),
            "normalization_cycles": float(
                normalization_cycles
            ),
            "accumulation_cycles": float(
                accumulation_cycles
            ),
            "pipeline_fill_cycles": float(
                pipeline_fill_cycles
            ),
            "predicted_cycles": float(total),
        }
    )

    return result


def _elementwise_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    result = _base_result(layer)
    unroll = _unroll(layer)

    elements = _output_elements(layer)
    element_unroll = _positive_int(
        unroll.get("element", 1)
    )
    pipeline_ii = _pipeline_ii(layer)
    element_blocks = _ceil_div(
        elements,
        element_unroll,
    )

    operation_latency = {
        "Relu": 2,
        "LeakyRelu": 4,
        "Add": 2,
        "Sigmoid": 10,
        "BatchNormalization": 6,
        "Flatten": 1,
        "Reshape": 1,
    }.get(
        str(layer.get("op_type", "")),
        2,
    )

    operation_latency = _latency_setting(
        raw_cfg,
        (
            str(
                layer.get(
                    "op_type",
                    "generic",
                )
            ).lower()
            + "_latency"
        ),
        operation_latency,
    )

    accumulation_cycles = (
        element_blocks
        * pipeline_ii
    )
    total = (
        accumulation_cycles
        + operation_latency
    )

    result.update(
        {
            "element_blocks": element_blocks,
            "accumulation_cycles": float(
                accumulation_cycles
            ),
            "pipeline_fill_cycles": float(
                operation_latency
            ),
            "predicted_cycles": float(total),
        }
    )

    return result


def estimate_architecture_layer_schedule(
    layer: Mapping[str, Any],
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    op_type = str(
        layer.get("op_type", "Unknown")
    )

    if op_type == "Dense":
        return _dense_schedule(
            layer,
            raw_cfg,
        )

    if op_type == "Conv":
        return _conv_schedule(
            layer,
            raw_cfg,
        )

    if op_type in {"MaxPool", "AvgPool"}:
        return _pool_schedule(
            layer,
            raw_cfg,
        )

    if op_type == "Softmax":
        return _softmax_schedule(
            layer,
            raw_cfg,
        )

    return _elementwise_schedule(
        layer,
        raw_cfg,
    )


def estimate_architecture_schedules(
    layers: Sequence[Mapping[str, Any]],
    raw_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for layer in layers:
        schedule = estimate_architecture_layer_schedule(
            layer,
            raw_cfg,
        )
        schedule.update(
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
            }
        )
        results.append(schedule)

    return results