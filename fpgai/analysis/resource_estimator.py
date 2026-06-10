from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from fpgai.numerics.precision_policy import (
    canonical_op_type,
    resolve_precision_for_op,
)


BRAM18_BITS = 18_432
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


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default

    return max(1, result)


def _positive_float(value: Any, default: float = 1.0) -> float:
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


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator <= 0:
        return 0

    return (numerator + max(1, denominator) - 1) // max(1, denominator)


def _shape_tuple(shape: Any) -> Tuple[int, ...]:
    if not isinstance(shape, (tuple, list)):
        return tuple()

    result: List[int] = []

    for dimension in shape:
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            return tuple()

        if value <= 0:
            return tuple()

        result.append(value)

    return tuple(result)


def _strip_batch(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]

    return shape


def _numel(shape: Sequence[int]) -> int:
    if not shape:
        return 0

    result = 1

    for dimension in shape:
        result *= int(dimension)

    return result


def _shape_at(shapes: Any, index: int) -> Tuple[int, ...]:
    if not isinstance(shapes, (list, tuple)):
        return tuple()

    if index < 0 or index >= len(shapes):
        return tuple()

    return _shape_tuple(shapes[index])


def _descriptor_op(descriptor: Any) -> SimpleNamespace:
    return SimpleNamespace(
        name=str(getattr(descriptor, "node_name", "")),
        op_type=str(getattr(descriptor, "op_type", "")),
        attrs=dict(getattr(descriptor, "attrs", {}) or {}),
    )


def _precision_for_descriptor(
    raw_cfg: Dict[str, Any],
    descriptor: Any,
    index: int,
) -> Dict[str, Dict[str, Any]]:
    return resolve_precision_for_op(
        raw_cfg,
        _descriptor_op(descriptor),
        index,
    ).specs


def _parallelism(
    raw_cfg: Dict[str, Any],
    op_type: str,
) -> Dict[str, int]:
    global_pe = _positive_int(
        _cfg_get(raw_cfg, "optimization.parallel.pe", 1)
    )
    global_simd = _positive_int(
        _cfg_get(raw_cfg, "optimization.parallel.simd", 1)
    )
    global_unroll = _positive_int(
        _cfg_get(raw_cfg, "optimization.parallel.unroll_factor", 1)
    )
    partition_factor = _positive_int(
        _cfg_get(raw_cfg, "optimization.parallel.partition_factor", 1)
    )

    if op_type == "Conv":
        pe = _positive_int(
            _cfg_get(raw_cfg, "hls.conv.oc_unroll", global_pe),
            global_pe,
        )
        simd = _positive_int(
            _cfg_get(raw_cfg, "hls.conv.ic_unroll", global_simd),
            global_simd,
        )
    elif op_type == "Dense":
        pe = _positive_int(
            _cfg_get(raw_cfg, "hls.dense.out_unroll", global_pe),
            global_pe,
        )
        simd = _positive_int(
            _cfg_get(raw_cfg, "hls.dense.in_unroll", global_simd),
            global_simd,
        )
    else:
        pe = global_unroll
        simd = 1

    return {
        "pe": pe,
        "simd": simd,
        "lanes": max(1, pe * simd),
        "partition_factor": partition_factor,
    }


def _dense_dimensions(descriptor: Any) -> Dict[str, int]:
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    input_shape = _strip_batch(
        _shape_at(getattr(descriptor, "input_shapes", []), 0)
    )
    weight_shape = _shape_at(
        getattr(descriptor, "input_shapes", []),
        1,
    )
    output_shape = _strip_batch(
        _shape_at(getattr(descriptor, "output_shapes", []), 0)
    )

    input_features = int(attrs.get("in_features", 0) or 0)
    output_features = int(attrs.get("out_features", 0) or 0)

    if input_features <= 0:
        input_features = _numel(input_shape)

    if output_features <= 0:
        output_features = _numel(output_shape)

    if len(weight_shape) == 2:
        if output_features <= 0:
            output_features = int(weight_shape[0])
        if input_features <= 0:
            input_features = int(weight_shape[1])

    weight_elements = max(0, input_features * output_features)
    bias_elements = max(0, output_features)

    return {
        "input_features": input_features,
        "output_features": output_features,
        "input_elements": input_features,
        "output_elements": output_features,
        "weight_elements": weight_elements,
        "bias_elements": bias_elements,
        "macs": weight_elements,
    }


def _conv_dimensions(descriptor: Any) -> Dict[str, int]:
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    input_shape = _strip_batch(
        _shape_at(getattr(descriptor, "input_shapes", []), 0)
    )
    weight_shape = _shape_at(
        getattr(descriptor, "input_shapes", []),
        1,
    )
    output_shape = _strip_batch(
        _shape_at(getattr(descriptor, "output_shapes", []), 0)
    )

    input_channels = 0
    input_height = 0
    input_width = 0
    output_channels = 0
    output_height = 0
    output_width = 0
    kernel_height = 0
    kernel_width = 0

    if len(input_shape) == 3:
        input_channels, input_height, input_width = input_shape

    if len(output_shape) == 3:
        output_channels, output_height, output_width = output_shape

    if len(weight_shape) == 4:
        weight_output_channels = int(weight_shape[0])
        weight_input_channels = int(weight_shape[1])
        kernel_height = int(weight_shape[2])
        kernel_width = int(weight_shape[3])

        if output_channels <= 0:
            output_channels = weight_output_channels
        if input_channels <= 0:
            input_channels = weight_input_channels

    kernel_shape = attrs.get("kernel_shape", [])

    if (
        (kernel_height <= 0 or kernel_width <= 0)
        and isinstance(kernel_shape, (list, tuple))
        and len(kernel_shape) >= 2
    ):
        kernel_height = int(kernel_shape[0])
        kernel_width = int(kernel_shape[1])

    if output_channels <= 0:
        output_channels = int(attrs.get("out_channels", 0) or 0)

    groups = max(1, int(attrs.get("group", 1) or 1))
    channels_per_group = _ceil_div(input_channels, groups)

    output_elements = (
        output_channels
        * output_height
        * output_width
    )
    weight_elements = (
        output_channels
        * channels_per_group
        * kernel_height
        * kernel_width
    )
    bias_elements = output_channels

    macs = (
        output_elements
        * channels_per_group
        * kernel_height
        * kernel_width
    )

    return {
        "input_channels": input_channels,
        "input_height": input_height,
        "input_width": input_width,
        "output_channels": output_channels,
        "output_height": output_height,
        "output_width": output_width,
        "kernel_height": kernel_height,
        "kernel_width": kernel_width,
        "groups": groups,
        "input_elements": input_channels * input_height * input_width,
        "output_elements": output_elements,
        "weight_elements": weight_elements,
        "bias_elements": bias_elements,
        "macs": max(0, macs),
    }


def _pool_dimensions(descriptor: Any) -> Dict[str, int]:
    attrs = dict(getattr(descriptor, "attrs", {}) or {})

    input_shape = _strip_batch(
        _shape_at(getattr(descriptor, "input_shapes", []), 0)
    )
    output_shape = _strip_batch(
        _shape_at(getattr(descriptor, "output_shapes", []), 0)
    )

    kernel_shape = attrs.get("kernel_shape", [1, 1])
    kernel_height = 1
    kernel_width = 1

    if isinstance(kernel_shape, (list, tuple)) and len(kernel_shape) >= 2:
        kernel_height = max(1, int(kernel_shape[0]))
        kernel_width = max(1, int(kernel_shape[1]))

    return {
        "input_elements": _numel(input_shape),
        "output_elements": _numel(output_shape),
        "kernel_height": kernel_height,
        "kernel_width": kernel_width,
        "kernel_elements": kernel_height * kernel_width,
        "macs": 0,
    }


def _operator_dimensions(descriptor: Any) -> Dict[str, int]:
    op_type = canonical_op_type(
        str(getattr(descriptor, "op_type", ""))
    )

    if op_type == "Dense":
        return _dense_dimensions(descriptor)

    if op_type == "Conv":
        return _conv_dimensions(descriptor)

    if op_type in {"MaxPool", "AvgPool"}:
        return _pool_dimensions(descriptor)

    input_shape = _strip_batch(
        _shape_at(getattr(descriptor, "input_shapes", []), 0)
    )
    output_shape = _strip_batch(
        _shape_at(getattr(descriptor, "output_shapes", []), 0)
    )

    return {
        "input_elements": _numel(input_shape),
        "output_elements": _numel(output_shape),
        "weight_elements": 0,
        "bias_elements": 0,
        "macs": int(getattr(descriptor, "macs", 0) or 0),
    }


def _multiplier_dsp_cost(
    left_bits: int,
    right_bits: int,
) -> int:
    if left_bits <= 0 or right_bits <= 0:
        return 0

    if left_bits <= 18 and right_bits <= 27:
        return 1

    if left_bits <= 27 and right_bits <= 18:
        return 1

    left_tiles = _ceil_div(left_bits, 18)
    right_tiles = _ceil_div(right_bits, 18)

    return max(1, left_tiles * right_tiles)


def _banked_bram18(
    elements: int,
    bits_per_element: int,
    banks: int = 1,
) -> int:
    if elements <= 0 or bits_per_element <= 0:
        return 0

    banks = min(max(1, banks), elements)
    elements_per_bank = _ceil_div(elements, banks)
    bits_per_bank = elements_per_bank * bits_per_element
    blocks_per_bank = _ceil_div(bits_per_bank, BRAM18_BITS)

    return banks * blocks_per_bank


def _memory_binding(
    raw_cfg: Dict[str, Any],
    key: str,
    default: str,
) -> str:
    value = _cfg_get(
        raw_cfg,
        f"memory.storage.{key}",
        default,
    )

    return str(value).strip().lower()


def _parameter_storage(
    *,
    dimensions: Mapping[str, int],
    parameter_bytes: int,
    weight_bits: int,
    bias_bits: int,
    banks: int,
    raw_cfg: Dict[str, Any],
) -> Tuple[int, int]:
    binding = _memory_binding(
        raw_cfg,
        "weights",
        "bram",
    )

    if binding not in {"bram", "uram", "auto"}:
        return 0, 0

    weight_elements = int(dimensions.get("weight_elements", 0) or 0)
    bias_elements = int(dimensions.get("bias_elements", 0) or 0)

    if weight_elements <= 0 and parameter_bytes > 0:
        weight_elements = _ceil_div(parameter_bytes, FLOAT_BYTES)

    weight_bram = _banked_bram18(
        weight_elements,
        weight_bits,
        banks,
    )
    bias_bram = _banked_bram18(
        bias_elements,
        bias_bits,
        min(banks, max(1, bias_elements)),
    )

    return weight_bram, bias_bram


def _activation_storage(
    *,
    output_elements: int,
    activation_bits: int,
    banks: int,
    raw_cfg: Dict[str, Any],
) -> int:
    binding = _memory_binding(
        raw_cfg,
        "activations",
        "bram",
    )

    if binding not in {"bram", "uram", "auto"}:
        return 0

    minimum_elements = _positive_int(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator.minimum_bram_elements",
            512,
        ),
        512,
    )

    if output_elements < minimum_elements:
        return 0

    return _banked_bram18(
        output_elements,
        activation_bits,
        banks,
    )


def _line_buffer_bram(
    dimensions: Mapping[str, int],
    activation_bits: int,
    banks: int,
) -> int:
    input_channels = int(dimensions.get("input_channels", 0) or 0)
    input_width = int(dimensions.get("input_width", 0) or 0)
    kernel_height = int(dimensions.get("kernel_height", 0) or 0)

    buffered_rows = max(0, kernel_height - 1)

    return _banked_bram18(
        buffered_rows * input_width * input_channels,
        activation_bits,
        banks,
    )


def _dense_resources(
    *,
    dimensions: Mapping[str, int],
    activation_bits: int,
    weight_bits: int,
    accumulator_bits: int,
    lanes: int,
) -> Dict[str, int]:
    multiplier_dsp_per_lane = _multiplier_dsp_cost(
        activation_bits,
        weight_bits,
    )
    dsp = lanes * multiplier_dsp_per_lane

    multiplier_logic = lanes * (
        20
        + 2 * activation_bits
        + 2 * weight_bits
    )
    accumulator_logic = lanes * (
        18
        + 3 * accumulator_bits
    )
    control_logic = 80 + int(
        12 * math.log2(max(2, int(dimensions.get("macs", 0)) + 1))
    )

    return {
        "lut": 180 + multiplier_logic + accumulator_logic + control_logic,
        "ff": (
            220
            + lanes
            * (
                30
                + 2 * activation_bits
                + 2 * weight_bits
                + 3 * accumulator_bits
            )
            + control_logic
        ),
        "dsp": dsp,
        "multiplier_dsp_per_lane": multiplier_dsp_per_lane,
    }


def _conv_resources(
    *,
    dimensions: Mapping[str, int],
    activation_bits: int,
    weight_bits: int,
    accumulator_bits: int,
    lanes: int,
) -> Dict[str, int]:
    multiplier_dsp_per_lane = _multiplier_dsp_cost(
        activation_bits,
        weight_bits,
    )
    dsp = lanes * multiplier_dsp_per_lane

    kernel_elements = (
        int(dimensions.get("kernel_height", 0) or 0)
        * int(dimensions.get("kernel_width", 0) or 0)
    )

    address_logic = (
        240
        + 35 * max(1, kernel_elements)
        + 12 * int(dimensions.get("input_channels", 0) or 0)
    )
    datapath_logic = lanes * (
        35
        + 3 * activation_bits
        + 3 * weight_bits
        + 4 * accumulator_bits
    )

    return {
        "lut": 420 + address_logic + datapath_logic,
        "ff": (
            520
            + address_logic
            + lanes
            * (
                45
                + 3 * activation_bits
                + 3 * weight_bits
                + 5 * accumulator_bits
            )
        ),
        "dsp": dsp,
        "multiplier_dsp_per_lane": multiplier_dsp_per_lane,
    }


def _elementwise_resources(
    *,
    op_type: str,
    activation_bits: int,
    accumulator_bits: int,
    lanes: int,
    dimensions: Mapping[str, int],
) -> Dict[str, int]:
    kernel_elements = int(dimensions.get("kernel_elements", 1) or 1)

    if op_type == "Relu":
        return {
            "lut": 35 + lanes * (activation_bits + 8),
            "ff": 40 + lanes * (activation_bits + 10),
            "dsp": 0,
            "multiplier_dsp_per_lane": 0,
        }

    if op_type == "LeakyRelu":
        dsp_per_lane = _multiplier_dsp_cost(
            activation_bits,
            activation_bits,
        )
        return {
            "lut": 60 + lanes * (2 * activation_bits + 18),
            "ff": 70 + lanes * (2 * activation_bits + 22),
            "dsp": lanes * dsp_per_lane,
            "multiplier_dsp_per_lane": dsp_per_lane,
        }

    if op_type == "Add":
        return {
            "lut": 45 + lanes * (activation_bits + 12),
            "ff": 55 + lanes * (activation_bits + 14),
            "dsp": 0,
            "multiplier_dsp_per_lane": 0,
        }

    if op_type == "MaxPool":
        comparators = max(1, kernel_elements - 1)
        return {
            "lut": 80 + lanes * comparators * (activation_bits + 4),
            "ff": 90 + lanes * comparators * (activation_bits + 6),
            "dsp": 0,
            "multiplier_dsp_per_lane": 0,
        }

    if op_type == "AvgPool":
        adders = max(1, kernel_elements - 1)
        return {
            "lut": (
                100
                + lanes
                * (
                    adders * (accumulator_bits + 4)
                    + activation_bits
                )
            ),
            "ff": (
                120
                + lanes
                * (
                    adders * (accumulator_bits + 6)
                    + activation_bits
                )
            ),
            "dsp": 0,
            "multiplier_dsp_per_lane": 0,
        }

    if op_type == "BatchNormalization":
        dsp_per_lane = 2 * _multiplier_dsp_cost(
            activation_bits,
            activation_bits,
        )
        return {
            "lut": 180 + lanes * (5 * activation_bits + 80),
            "ff": 220 + lanes * (6 * activation_bits + 95),
            "dsp": lanes * dsp_per_lane,
            "multiplier_dsp_per_lane": dsp_per_lane,
        }

    if op_type == "Sigmoid":
        return {
            "lut": 500 + lanes * (18 * activation_bits + 160),
            "ff": 560 + lanes * (20 * activation_bits + 180),
            "dsp": lanes * 3,
            "multiplier_dsp_per_lane": 3,
        }

    if op_type == "Softmax":
        return {
            "lut": 900 + lanes * (24 * activation_bits + 240),
            "ff": 1000 + lanes * (26 * activation_bits + 280),
            "dsp": lanes * 8,
            "multiplier_dsp_per_lane": 8,
        }

    if op_type in {"Flatten", "Reshape"}:
        return {
            "lut": 20,
            "ff": 20,
            "dsp": 0,
            "multiplier_dsp_per_lane": 0,
        }

    return {
        "lut": 100 + lanes * (activation_bits + 20),
        "ff": 120 + lanes * (activation_bits + 24),
        "dsp": 0,
        "multiplier_dsp_per_lane": 0,
    }


def _explicit_calibration(
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    configured = _cfg_get(
        raw_cfg,
        "analysis.design_space.calibration.resources",
        {},
    )

    if not isinstance(configured, Mapping) or not configured:
        return {
            "enabled": False,
            "fixed_lut": 0.0,
            "fixed_ff": 0.0,
            "fixed_dsp": 0.0,
            "fixed_bram18": 0.0,
            "lut_scale": 1.0,
            "ff_scale": 1.0,
            "dsp_scale": 1.0,
            "bram18_scale": 1.0,
        }

    explicitly_enabled = configured.get("enabled")
    enabled = (
        bool(explicitly_enabled)
        if explicitly_enabled is not None
        else True
    )

    return {
        "enabled": enabled,
        "fixed_lut": _nonnegative_float(
            configured.get("fixed_lut", 0.0)
        ),
        "fixed_ff": _nonnegative_float(
            configured.get("fixed_ff", 0.0)
        ),
        "fixed_dsp": _nonnegative_float(
            configured.get("fixed_dsp", 0.0)
        ),
        "fixed_bram18": _nonnegative_float(
            configured.get("fixed_bram18", 0.0)
        ),
        "lut_scale": _positive_float(
            configured.get("lut_scale", 1.0)
        ),
        "ff_scale": _positive_float(
            configured.get("ff_scale", 1.0)
        ),
        "dsp_scale": _positive_float(
            configured.get("dsp_scale", 1.0)
        ),
        "bram18_scale": _positive_float(
            configured.get("bram18_scale", 1.0)
        ),
    }


def _estimate_layer_resources(
    descriptor: Any,
    index: int,
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    op_type = canonical_op_type(
        str(getattr(descriptor, "op_type", "Unknown"))
    )
    precision = _precision_for_descriptor(
        raw_cfg,
        descriptor,
        index,
    )
    parallelism = _parallelism(
        raw_cfg,
        op_type,
    )
    dimensions = _operator_dimensions(descriptor)

    activation_bits = int(
        precision["activation"]["total_bits"]
    )
    weight_bits = int(
        precision["weight"]["total_bits"]
    )
    bias_bits = int(
        precision["bias"]["total_bits"]
    )
    accumulator_bits = int(
        precision["accum"]["total_bits"]
    )

    parameter_bytes = int(
        getattr(descriptor, "param_bytes", 0) or 0
    )
    activation_bytes_in = int(
        getattr(descriptor, "activation_bytes_in", 0) or 0
    )
    activation_bytes_out = int(
        getattr(descriptor, "activation_bytes_out", 0) or 0
    )

    macs = int(dimensions.get("macs", 0) or 0)

    if macs <= 0:
        macs = int(getattr(descriptor, "macs", 0) or 0)

    lanes = parallelism["lanes"]

    if op_type == "Dense":
        arithmetic = _dense_resources(
            dimensions=dimensions,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            accumulator_bits=accumulator_bits,
            lanes=lanes,
        )
    elif op_type == "Conv":
        arithmetic = _conv_resources(
            dimensions=dimensions,
            activation_bits=activation_bits,
            weight_bits=weight_bits,
            accumulator_bits=accumulator_bits,
            lanes=lanes,
        )
    else:
        arithmetic = _elementwise_resources(
            op_type=op_type,
            activation_bits=activation_bits,
            accumulator_bits=accumulator_bits,
            lanes=lanes,
            dimensions=dimensions,
        )

    parameter_bram, bias_bram = _parameter_storage(
        dimensions=dimensions,
        parameter_bytes=parameter_bytes,
        weight_bits=weight_bits,
        bias_bits=bias_bits,
        banks=max(
            parallelism["partition_factor"],
            parallelism["pe"],
        ),
        raw_cfg=raw_cfg,
    )

    output_elements = int(
        dimensions.get("output_elements", 0) or 0
    )

    if output_elements <= 0:
        output_elements = _ceil_div(
            activation_bytes_out,
            FLOAT_BYTES,
        )

    activation_bram = _activation_storage(
        output_elements=output_elements,
        activation_bits=activation_bits,
        banks=parallelism["partition_factor"],
        raw_cfg=raw_cfg,
    )

    line_buffer_bram = 0

    if op_type in {"Conv", "MaxPool", "AvgPool"}:
        line_buffer_bram = _line_buffer_bram(
            dimensions,
            activation_bits,
            parallelism["partition_factor"],
        )

    bram18 = (
        parameter_bram
        + bias_bram
        + activation_bram
        + line_buffer_bram
    )

    return {
        "layer_index": index,
        "layer_name": str(
            getattr(descriptor, "node_name", f"layer_{index}")
        ),
        "op_type": op_type,
        "activation_bits": activation_bits,
        "weight_bits": weight_bits,
        "bias_bits": bias_bits,
        "accumulator_bits": accumulator_bits,
        "act_bits": activation_bits,
        "wgt_bits": weight_bits,
        "acc_bits": accumulator_bits,
        "pe": parallelism["pe"],
        "simd": parallelism["simd"],
        "partition_factor": parallelism["partition_factor"],
        "multiplier_lanes": lanes,
        "multiplier_dsp_per_lane": arithmetic[
            "multiplier_dsp_per_lane"
        ],
        "macs": macs,
        "parameter_bytes": parameter_bytes,
        "param_bytes": parameter_bytes,
        "activation_bytes_in": activation_bytes_in,
        "activation_bytes_out": activation_bytes_out,
        "attrs": dict(getattr(descriptor, "attrs", {}) or {}),
        "dimensions": dimensions,
        "resource_components": {
            "arithmetic_lut": int(arithmetic["lut"]),
            "arithmetic_ff": int(arithmetic["ff"]),
            "arithmetic_dsp": int(arithmetic["dsp"]),
            "parameter_bram18": parameter_bram,
            "bias_bram18": bias_bram,
            "activation_bram18": activation_bram,
            "line_buffer_bram18": line_buffer_bram,
        },
        "predicted_lut_raw": int(arithmetic["lut"]),
        "predicted_ff_raw": int(arithmetic["ff"]),
        "predicted_dsp_raw": int(arithmetic["dsp"]),
        "predicted_bram18_raw": int(bram18),
        "predicted_lut": int(arithmetic["lut"]),
        "predicted_ff": int(arithmetic["ff"]),
        "predicted_dsp": int(arithmetic["dsp"]),
        "predicted_bram18": int(bram18),
    }


def _top_level_resources(
    layers: List[Dict[str, Any]],
    raw_cfg: Dict[str, Any],
) -> Dict[str, int]:
    layer_count = len(layers)
    stream_edges = max(0, layer_count + 1)

    input_elements = 0
    output_elements = 0
    activation_bits = 16

    if layers:
        first = layers[0]
        last = layers[-1]

        input_elements = int(
            first["dimensions"].get("input_elements", 0) or 0
        )
        output_elements = int(
            last["dimensions"].get("output_elements", 0) or 0
        )
        activation_bits = int(first["activation_bits"])

    interface_lut = 260 + 45 * stream_edges
    interface_ff = 340 + 64 * stream_edges

    dispatch_lut = 70 * layer_count
    dispatch_ff = 90 * layer_count

    conversion_lut = (
        _ceil_div(input_elements, 64)
        + _ceil_div(output_elements, 64)
    )
    conversion_ff = (
        _ceil_div(input_elements * activation_bits, 128)
        + _ceil_div(output_elements * activation_bits, 128)
    )

    io_bram = 0

    if bool(
        _cfg_get(
            raw_cfg,
            "analysis.design_space.estimator.buffer_top_io",
            False,
        )
    ):
        io_bram = (
            _banked_bram18(
                input_elements,
                activation_bits,
            )
            + _banked_bram18(
                output_elements,
                activation_bits,
            )
        )

    return {
        "predicted_lut": (
            interface_lut
            + dispatch_lut
            + conversion_lut
        ),
        "predicted_ff": (
            interface_ff
            + dispatch_ff
            + conversion_ff
        ),
        "predicted_dsp": 0,
        "predicted_bram18": io_bram,
        "interface_lut": interface_lut,
        "interface_ff": interface_ff,
        "dispatch_lut": dispatch_lut,
        "dispatch_ff": dispatch_ff,
        "conversion_lut": conversion_lut,
        "conversion_ff": conversion_ff,
        "io_bram18": io_bram,
    }


def _apply_explicit_calibration(
    *,
    raw_value: int,
    fixed_value: float,
    scale: float,
    enabled: bool,
) -> int:
    if not enabled:
        return max(0, int(raw_value))

    return max(
        0,
        int(round((raw_value + fixed_value) * scale)),
    )


def estimate_resources_from_descriptors(
    descriptors: List[Any],
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    layers = [
        _estimate_layer_resources(
            descriptor,
            index,
            raw_cfg,
        )
        for index, descriptor in enumerate(descriptors)
    ]

    top_level = _top_level_resources(
        layers,
        raw_cfg,
    )
    calibration = _explicit_calibration(raw_cfg)

    raw_lut = (
        sum(layer["predicted_lut_raw"] for layer in layers)
        + top_level["predicted_lut"]
    )
    raw_ff = (
        sum(layer["predicted_ff_raw"] for layer in layers)
        + top_level["predicted_ff"]
    )
    raw_dsp = (
        sum(layer["predicted_dsp_raw"] for layer in layers)
        + top_level["predicted_dsp"]
    )
    raw_bram18 = (
        sum(layer["predicted_bram18_raw"] for layer in layers)
        + top_level["predicted_bram18"]
    )

    predicted_lut = _apply_explicit_calibration(
        raw_value=raw_lut,
        fixed_value=calibration["fixed_lut"],
        scale=calibration["lut_scale"],
        enabled=calibration["enabled"],
    )
    predicted_ff = _apply_explicit_calibration(
        raw_value=raw_ff,
        fixed_value=calibration["fixed_ff"],
        scale=calibration["ff_scale"],
        enabled=calibration["enabled"],
    )
    predicted_dsp = _apply_explicit_calibration(
        raw_value=raw_dsp,
        fixed_value=calibration["fixed_dsp"],
        scale=calibration["dsp_scale"],
        enabled=calibration["enabled"],
    )
    predicted_bram18 = _apply_explicit_calibration(
        raw_value=raw_bram18,
        fixed_value=calibration["fixed_bram18"],
        scale=calibration["bram18_scale"],
        enabled=calibration["enabled"],
    )

    totals = {
        "predicted_lut": predicted_lut,
        "predicted_ff": predicted_ff,
        "predicted_dsp": predicted_dsp,
        "predicted_bram18": predicted_bram18,
        "predicted_lut_raw": int(raw_lut),
        "predicted_ff_raw": int(raw_ff),
        "predicted_dsp_raw": int(raw_dsp),
        "predicted_bram18_raw": int(raw_bram18),
        "total_macs": int(
            sum(layer["macs"] for layer in layers)
        ),
        "total_multiplier_lanes": int(
            sum(
                layer["multiplier_lanes"]
                for layer in layers
                if layer["op_type"] in {"Dense", "Conv"}
            )
        ),
    }

    return {
        # Retained so existing report readers and tests remain compatible.
        "model": "operator_aware_calibrated_v1",
        "analytical_model": "operator_structural_v2",
        "estimation_mode": (
            "analytical_with_explicit_calibration"
            if calibration["enabled"]
            else "analytical"
        ),
        "calibration": calibration,
        "top_level": top_level,
        "totals": totals,
        "worst_lut_layer": (
            max(
                layers,
                key=lambda layer: layer["predicted_lut"],
            )
            if layers
            else None
        ),
        "worst_dsp_layer": (
            max(
                layers,
                key=lambda layer: layer["predicted_dsp"],
            )
            if layers
            else None
        ),
        "worst_bram_layer": (
            max(
                layers,
                key=lambda layer: layer["predicted_bram18"],
            )
            if layers
            else None
        ),
        "layers": layers,
    }