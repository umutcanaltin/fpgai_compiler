from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Tuple

from fpgai.numerics.precision_policy import (
    canonical_op_type,
    resolve_precision_for_op,
)


BRAM18_BITS = 18_432

DEFAULT_CALIBRATION = {
    # Calibrated from the current KV260/Vitis HLS 2023.2 result.
    # These remain configurable under:
    # analysis.design_space.calibration.resources
    "fixed_lut": 7_500.0,
    "fixed_ff": 8_000.0,
    "fixed_dsp": 14.0,
    "fixed_bram18": 2.0,
    "lut_scale": 1.0,
    "ff_scale": 1.0,
    "dsp_scale": 1.0,
    "bram18_scale": 1.0,
}


OPERATOR_BASE_COST = {
    "Dense": {
        "lut": 420.0,
        "ff": 500.0,
        "dsp": 0.0,
    },
    "Conv": {
        "lut": 700.0,
        "ff": 850.0,
        "dsp": 0.0,
    },
    "MaxPool": {
        "lut": 220.0,
        "ff": 260.0,
        "dsp": 0.0,
    },
    "AvgPool": {
        "lut": 300.0,
        "ff": 340.0,
        "dsp": 1.0,
    },
    "Relu": {
        "lut": 90.0,
        "ff": 100.0,
        "dsp": 0.0,
    },
    "LeakyRelu": {
        "lut": 130.0,
        "ff": 150.0,
        "dsp": 1.0,
    },
    "Sigmoid": {
        "lut": 500.0,
        "ff": 540.0,
        "dsp": 3.0,
    },
    "Softmax": {
        "lut": 1_100.0,
        "ff": 1_200.0,
        "dsp": 8.0,
    },
    "BatchNormalization": {
        "lut": 550.0,
        "ff": 650.0,
        "dsp": 3.0,
    },
    "Add": {
        "lut": 120.0,
        "ff": 140.0,
        "dsp": 0.0,
    },
    "Flatten": {
        "lut": 20.0,
        "ff": 20.0,
        "dsp": 0.0,
    },
    "Reshape": {
        "lut": 40.0,
        "ff": 40.0,
        "dsp": 0.0,
    },
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


def _positive_int(
    value: Any,
    default: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default

    return max(1, result)


def _positive_float(
    value: Any,
    default: float,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result) or result <= 0.0:
        return default

    return result


def _shape_tuple(
    shape: Any,
) -> Tuple[int, ...]:
    if not isinstance(
        shape,
        (
            tuple,
            list,
        ),
    ):
        return tuple()

    result = []

    for dimension in shape:
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            return tuple()

        if value <= 0:
            return tuple()

        result.append(value)

    return tuple(result)


def _strip_batch(
    shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    if (
        len(shape) > 1
        and shape[0] == 1
    ):
        return shape[1:]

    return shape


def _numel(
    shape: Tuple[int, ...],
) -> int:
    if not shape:
        return 0

    result = 1

    for dimension in shape:
        result *= int(dimension)

    return result


def _first_shape(
    shapes: Any,
) -> Tuple[int, ...]:
    if not isinstance(shapes, list) or not shapes:
        return tuple()

    return _shape_tuple(shapes[0])


def _second_shape(
    shapes: Any,
) -> Tuple[int, ...]:
    if not isinstance(shapes, list) or len(shapes) < 2:
        return tuple()

    return _shape_tuple(shapes[1])


def _descriptor_op(
    descriptor: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=str(
            getattr(
                descriptor,
                "node_name",
                "",
            )
        ),
        op_type=str(
            getattr(
                descriptor,
                "op_type",
                "",
            )
        ),
        attrs=dict(
            getattr(
                descriptor,
                "attrs",
                {},
            )
            or {}
        ),
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
        _cfg_get(
            raw_cfg,
            "optimization.parallel.pe",
            1,
        ),
        1,
    )
    global_simd = _positive_int(
        _cfg_get(
            raw_cfg,
            "optimization.parallel.simd",
            1,
        ),
        1,
    )
    global_unroll = _positive_int(
        _cfg_get(
            raw_cfg,
            "optimization.parallel.unroll_factor",
            1,
        ),
        1,
    )
    partition_factor = _positive_int(
        _cfg_get(
            raw_cfg,
            "optimization.parallel.partition_factor",
            1,
        ),
        1,
    )

    if op_type == "Conv":
        pe = _positive_int(
            _cfg_get(
                raw_cfg,
                "hls.conv.oc_unroll",
                global_pe,
            ),
            global_pe,
        )
        simd = _positive_int(
            _cfg_get(
                raw_cfg,
                "hls.conv.ic_unroll",
                global_simd,
            ),
            global_simd,
        )
    elif op_type == "Dense":
        pe = _positive_int(
            _cfg_get(
                raw_cfg,
                "hls.dense.out_unroll",
                global_pe,
            ),
            global_pe,
        )
        simd = _positive_int(
            _cfg_get(
                raw_cfg,
                "hls.dense.in_unroll",
                global_simd,
            ),
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


def _dense_dimensions(
    descriptor: Any,
) -> Dict[str, int]:
    attrs = dict(
        getattr(
            descriptor,
            "attrs",
            {},
        )
        or {}
    )

    input_features = int(
        attrs.get(
            "in_features",
            0,
        )
        or 0
    )
    output_features = int(
        attrs.get(
            "out_features",
            0,
        )
        or 0
    )

    input_shape = _strip_batch(
        _first_shape(
            getattr(
                descriptor,
                "input_shapes",
                [],
            )
        )
    )
    output_shape = _strip_batch(
        _first_shape(
            getattr(
                descriptor,
                "output_shapes",
                [],
            )
        )
    )
    weight_shape = _second_shape(
        getattr(
            descriptor,
            "input_shapes",
            [],
        )
    )

    if input_features <= 0:
        input_features = _numel(input_shape)

    if output_features <= 0:
        output_features = _numel(output_shape)

    if (
        (
            input_features <= 0
            or output_features <= 0
        )
        and len(weight_shape) == 2
    ):
        output_features = int(
            weight_shape[0]
        )
        input_features = int(
            weight_shape[1]
        )

    macs = max(
        0,
        input_features * output_features,
    )

    return {
        "input_features": input_features,
        "output_features": output_features,
        "macs": macs,
    }


def _conv_dimensions(
    descriptor: Any,
) -> Dict[str, int]:
    attrs = dict(
        getattr(
            descriptor,
            "attrs",
            {},
        )
        or {}
    )

    input_shape = _first_shape(
        getattr(
            descriptor,
            "input_shapes",
            [],
        )
    )
    output_shape = _first_shape(
        getattr(
            descriptor,
            "output_shapes",
            [],
        )
    )
    weight_shape = _second_shape(
        getattr(
            descriptor,
            "input_shapes",
            [],
        )
    )

    input_shape = _strip_batch(input_shape)
    output_shape = _strip_batch(output_shape)

    input_channels = 0
    input_height = 0
    input_width = 0
    output_channels = 0
    output_height = 0
    output_width = 0
    kernel_height = 0
    kernel_width = 0

    if len(input_shape) == 3:
        (
            input_channels,
            input_height,
            input_width,
        ) = input_shape

    if len(output_shape) == 3:
        (
            output_channels,
            output_height,
            output_width,
        ) = output_shape

    if len(weight_shape) == 4:
        output_channels = int(
            weight_shape[0]
        )
        input_channels = int(
            weight_shape[1]
        )
        kernel_height = int(
            weight_shape[2]
        )
        kernel_width = int(
            weight_shape[3]
        )

    kernel_shape = attrs.get(
        "kernel_shape",
        [],
    )

    if (
        (
            kernel_height <= 0
            or kernel_width <= 0
        )
        and isinstance(kernel_shape, list)
        and len(kernel_shape) >= 2
    ):
        kernel_height = int(
            kernel_shape[0]
        )
        kernel_width = int(
            kernel_shape[1]
        )

    if output_channels <= 0:
        output_channels = int(
            attrs.get(
                "out_channels",
                0,
            )
            or 0
        )

    macs = (
        output_height
        * output_width
        * output_channels
        * input_channels
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
        "macs": max(0, macs),
    }


def _operator_dimensions(
    descriptor: Any,
) -> Dict[str, int]:
    op_type = canonical_op_type(
        str(
            getattr(
                descriptor,
                "op_type",
                "",
            )
        )
    )

    if op_type == "Dense":
        return _dense_dimensions(
            descriptor
        )

    if op_type == "Conv":
        return _conv_dimensions(
            descriptor
        )

    input_shape = _strip_batch(
        _first_shape(
            getattr(
                descriptor,
                "input_shapes",
                [],
            )
        )
    )
    output_shape = _strip_batch(
        _first_shape(
            getattr(
                descriptor,
                "output_shapes",
                [],
            )
        )
    )

    return {
        "input_elements": _numel(
            input_shape
        ),
        "output_elements": _numel(
            output_shape
        ),
        "macs": int(
            getattr(
                descriptor,
                "macs",
                0,
            )
            or 0
        ),
    }


def _multiplier_dsp_cost(
    activation_bits: int,
    weight_bits: int,
) -> int:
    # Conservative DSP48 approximation.
    # Small products may share one DSP; wider products
    # require multiple DSP slices.
    if (
        activation_bits <= 8
        and weight_bits <= 8
    ):
        return 1

    if (
        activation_bits <= 18
        and weight_bits <= 18
    ):
        return 1

    activation_tiles = math.ceil(
        activation_bits / 18
    )
    weight_tiles = math.ceil(
        weight_bits / 18
    )

    return max(
        1,
        activation_tiles * weight_tiles,
    )


def _storage_bram18(
    *,
    parameter_bytes: int,
    activation_bytes_in: int,
    activation_bytes_out: int,
    weight_bits: int,
    activation_bits: int,
    partition_factor: int,
) -> int:
    parameter_values = math.ceil(
        parameter_bytes / 4
    )
    input_values = math.ceil(
        activation_bytes_in / 4
    )
    output_values = math.ceil(
        activation_bytes_out / 4
    )

    parameter_bits = (
        parameter_values
        * weight_bits
    )
    activation_bits_total = (
        (input_values + output_values)
        * activation_bits
    )

    raw_blocks = math.ceil(
        (
            parameter_bits
            + activation_bits_total
        )
        / BRAM18_BITS
    )

    if raw_blocks == 0:
        return 0

    banking_overhead = max(
        0,
        partition_factor - 1,
    )

    return raw_blocks + banking_overhead


def _calibration(
    raw_cfg: Dict[str, Any],
) -> Dict[str, float]:
    result = dict(
        DEFAULT_CALIBRATION
    )

    configured = _cfg_get(
        raw_cfg,
        "analysis.design_space.calibration.resources",
        {},
    )

    if isinstance(configured, Mapping):
        for key in result:
            if key in configured:
                result[key] = _positive_float(
                    configured[key],
                    result[key],
                )

    return result


def _estimate_layer_resources(
    descriptor: Any,
    index: int,
    raw_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    op_type = canonical_op_type(
        str(
            getattr(
                descriptor,
                "op_type",
                "Unknown",
            )
        )
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
    dimensions = _operator_dimensions(
        descriptor
    )

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
        getattr(
            descriptor,
            "param_bytes",
            0,
        )
        or 0
    )
    activation_bytes_in = int(
        getattr(
            descriptor,
            "activation_bytes_in",
            0,
        )
        or 0
    )
    activation_bytes_out = int(
        getattr(
            descriptor,
            "activation_bytes_out",
            0,
        )
        or 0
    )

    macs = int(
        dimensions.get(
            "macs",
            0,
        )
    )

    if macs <= 0:
        macs = int(
            getattr(
                descriptor,
                "macs",
                0,
            )
            or 0
        )

    base = OPERATOR_BASE_COST.get(
        op_type,
        {
            "lut": 180.0,
            "ff": 220.0,
            "dsp": 0.0,
        },
    )

    lanes = parallelism["lanes"]
    multiplier_dsp = 0

    if op_type in {
        "Dense",
        "Conv",
    }:
        multiplier_dsp = (
            lanes
            * _multiplier_dsp_cost(
                activation_bits,
                weight_bits,
            )
        )

    arithmetic_width = (
        activation_bits
        + weight_bits
        + accumulator_bits
    )
    width_ratio = (
        arithmetic_width / 48.0
    )

    lane_lut = (
        lanes
        * (
            115.0
            + 7.0 * activation_bits
            + 7.0 * weight_bits
            + 4.0 * accumulator_bits
        )
    )
    lane_ff = (
        lanes
        * (
            150.0
            + 8.0 * activation_bits
            + 8.0 * weight_bits
            + 5.0 * accumulator_bits
        )
    )

    control_complexity = (
        55.0
        * math.log2(
            max(2, macs + 1)
        )
    )

    storage_lut = (
        0.035
        * (
            activation_bytes_in
            + activation_bytes_out
        )
    )
    storage_ff = (
        0.045
        * (
            activation_bytes_in
            + activation_bytes_out
        )
    )

    predicted_lut_raw = int(
        math.ceil(
            base["lut"]
            + lane_lut
            + control_complexity
            + storage_lut
            + 80.0 * width_ratio
        )
    )
    predicted_ff_raw = int(
        math.ceil(
            base["ff"]
            + lane_ff
            + 1.15 * control_complexity
            + storage_ff
            + 100.0 * width_ratio
        )
    )
    predicted_dsp_raw = int(
        math.ceil(
            base["dsp"]
            + multiplier_dsp
        )
    )
    predicted_bram18_raw = (
        _storage_bram18(
            parameter_bytes=parameter_bytes,
            activation_bytes_in=(
                activation_bytes_in
            ),
            activation_bytes_out=(
                activation_bytes_out
            ),
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            partition_factor=parallelism[
                "partition_factor"
            ],
        )
    )

    return {
        "layer_index": index,
        "layer_name": str(
            getattr(
                descriptor,
                "node_name",
                f"layer_{index}",
            )
        ),
        "op_type": op_type,
        "activation_bits": activation_bits,
        "weight_bits": weight_bits,
        "bias_bits": bias_bits,
        "accumulator_bits": accumulator_bits,
        "pe": parallelism["pe"],
        "simd": parallelism["simd"],
        "multiplier_lanes": lanes,
        "multiplier_dsp_per_lane": (
            _multiplier_dsp_cost(
                activation_bits,
                weight_bits,
            )
            if op_type in {
                "Dense",
                "Conv",
            }
            else 0
        ),
        "macs": macs,
        "parameter_bytes": parameter_bytes,
        "activation_bytes_in": (
            activation_bytes_in
        ),
        "activation_bytes_out": (
            activation_bytes_out
        ),
        "dimensions": dimensions,
        "predicted_lut_raw": (
            predicted_lut_raw
        ),
        "predicted_ff_raw": (
            predicted_ff_raw
        ),
        "predicted_dsp_raw": (
            predicted_dsp_raw
        ),
        "predicted_bram18_raw": (
            predicted_bram18_raw
        ),
        # Per-layer values intentionally exclude
        # top-level fixed overhead.
        "predicted_lut": predicted_lut_raw,
        "predicted_ff": predicted_ff_raw,
        "predicted_dsp": predicted_dsp_raw,
        "predicted_bram18": (
            predicted_bram18_raw
        ),
    }


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
        for index, descriptor in enumerate(
            descriptors
        )
    ]

    calibration = _calibration(
        raw_cfg
    )

    raw_lut = sum(
        layer["predicted_lut_raw"]
        for layer in layers
    )
    raw_ff = sum(
        layer["predicted_ff_raw"]
        for layer in layers
    )
    raw_dsp = sum(
        layer["predicted_dsp_raw"]
        for layer in layers
    )
    raw_bram18 = sum(
        layer["predicted_bram18_raw"]
        for layer in layers
    )

    predicted_lut = int(
        round(
            (
                calibration["fixed_lut"]
                + raw_lut
            )
            * calibration["lut_scale"]
        )
    )
    predicted_ff = int(
        round(
            (
                calibration["fixed_ff"]
                + raw_ff
            )
            * calibration["ff_scale"]
        )
    )
    predicted_dsp = int(
        round(
            (
                calibration["fixed_dsp"]
                + raw_dsp
            )
            * calibration["dsp_scale"]
        )
    )
    predicted_bram18 = int(
        round(
            (
                calibration["fixed_bram18"]
                + raw_bram18
            )
            * calibration["bram18_scale"]
        )
    )

    totals = {
        "predicted_lut": max(
            0,
            predicted_lut,
        ),
        "predicted_ff": max(
            0,
            predicted_ff,
        ),
        "predicted_dsp": max(
            0,
            predicted_dsp,
        ),
        "predicted_bram18": max(
            0,
            predicted_bram18,
        ),
        "predicted_lut_raw": int(
            raw_lut
        ),
        "predicted_ff_raw": int(
            raw_ff
        ),
        "predicted_dsp_raw": int(
            raw_dsp
        ),
        "predicted_bram18_raw": int(
            raw_bram18
        ),
        "total_macs": int(
            sum(
                layer["macs"]
                for layer in layers
            )
        ),
        "total_multiplier_lanes": int(
            sum(
                layer["multiplier_lanes"]
                for layer in layers
                if layer["op_type"] in {
                    "Dense",
                    "Conv",
                }
            )
        ),
    }

    return {
        "model": (
            "operator_aware_calibrated_v1"
        ),
        "calibration": calibration,
        "totals": totals,
        "worst_lut_layer": (
            max(
                layers,
                key=lambda layer: (
                    layer["predicted_lut"]
                ),
            )
            if layers
            else None
        ),
        "worst_dsp_layer": (
            max(
                layers,
                key=lambda layer: (
                    layer["predicted_dsp"]
                ),
            )
            if layers
            else None
        ),
        "worst_bram_layer": (
            max(
                layers,
                key=lambda layer: (
                    layer["predicted_bram18"]
                ),
            )
            if layers
            else None
        ),
        "layers": layers,
    }