from __future__ import annotations

import math
from typing import Any, Mapping

from fpgai.analysis.hls_architecture import (
    HLSArchitecture,
    LayerArchitecture,
)


BRAM18_BITS = 18_432


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0

    divisor = max(1, divisor)
    return (value + divisor - 1) // divisor


def _multiplier_dsp_cost(
    left_bits: int,
    right_bits: int,
    *,
    op_type: str,
) -> int:
    if left_bits <= 0 or right_bits <= 0:
        return 0

    if left_bits <= 18 and right_bits <= 27:
        return 1

    if left_bits <= 27 and right_bits <= 18:
        return 1

    if op_type == "Dense":
        forward = (
            _ceil_div(left_bits, 27)
            * _ceil_div(right_bits, 18)
        )
        reverse = (
            _ceil_div(left_bits, 18)
            * _ceil_div(right_bits, 27)
        )
        return max(1, min(forward, reverse))

    # Conv pipelines a complete reduction body. Vitis commonly creates
    # wider independent multiplier structures for these overlapped stages.
    return max(
        1,
        _ceil_div(left_bits, 18)
        * _ceil_div(right_bits, 18),
    )


def _banked_bram18(
    elements: int,
    bits_per_element: int,
    banks: int,
) -> int:
    if elements <= 0 or bits_per_element <= 0:
        return 0

    banks = min(
        max(1, banks),
        elements,
    )
    elements_per_bank = _ceil_div(
        elements,
        banks,
    )
    bits_per_bank = (
        elements_per_bank
        * bits_per_element
    )

    return banks * _ceil_div(
        bits_per_bank,
        BRAM18_BITS,
    )


def _memory_binding(
    raw_cfg: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    current: Any = raw_cfg

    for part in f"memory.storage.{key}".split("."):
        if (
            not isinstance(current, Mapping)
            or part not in current
        ):
            return default

        current = current[part]

    return str(current).strip().lower()


def _parameter_bram(
    layer: LayerArchitecture,
    raw_cfg: Mapping[str, Any],
) -> tuple[int, int]:
    if (
        _memory_binding(
            raw_cfg,
            "weights",
            "bram",
        )
        not in {"bram", "auto"}
    ):
        return 0, 0

    dimensions = layer.dimensions
    arithmetic = layer.arithmetic
    memory = layer.memory

    weight_elements = int(
        dimensions.get("weight_elements", 0)
    )
    bias_elements = int(
        dimensions.get("bias_elements", 0)
    )

    weight_bram = _banked_bram18(
        weight_elements,
        int(arithmetic["weight_bits"]),
        int(memory["weight_banks"]),
    )
    bias_bram = _banked_bram18(
        bias_elements,
        int(arithmetic["bias_bits"]),
        int(memory["output_banks"]),
    )

    return weight_bram, bias_bram


def _activation_bram(
    layer: LayerArchitecture,
    raw_cfg: Mapping[str, Any],
) -> int:
    if (
        _memory_binding(
            raw_cfg,
            "activations",
            "bram",
        )
        not in {"bram", "auto"}
    ):
        return 0

    output_elements = int(
        layer.dimensions.get(
            "output_elements",
            0,
        )
    )

    if output_elements < 512:
        return 0

    return _banked_bram18(
        output_elements,
        int(layer.arithmetic["output_bits"]),
        int(layer.memory["output_banks"]),
    )


def _line_buffer_bram(
    layer: LayerArchitecture,
) -> int:
    if layer.op_type not in {
        "Conv",
        "MaxPool",
        "AvgPool",
    }:
        return 0

    dimensions = layer.dimensions

    input_channels = int(
        dimensions.get("input_channels", 0)
    )
    input_width = int(
        dimensions.get("input_width", 0)
    )
    kernel_height = int(
        dimensions.get("kernel_height", 1)
    )

    buffered_rows = max(
        0,
        kernel_height - 1,
    )

    return _banked_bram18(
        buffered_rows
        * input_width
        * input_channels,
        int(layer.arithmetic["activation_bits"]),
        int(layer.memory["input_banks"]),
    )


def _dense_logic(
    layer: LayerArchitecture,
) -> tuple[int, int]:
    arithmetic = layer.arithmetic
    lanes = layer.explicit_lanes

    activation_bits = int(
        arithmetic["activation_bits"]
    )
    weight_bits = int(
        arithmetic["weight_bits"]
    )
    accumulator_bits = int(
        arithmetic["accumulator_bits"]
    )

    lut = (
        180
        + lanes
        * (
            30
            + 2 * activation_bits
            + 2 * weight_bits
            + 3 * accumulator_bits
        )
    )
    ff = (
        220
        + lanes
        * (
            38
            + 2 * activation_bits
            + 2 * weight_bits
            + 4 * accumulator_bits
        )
    )

    return lut, ff


def _conv_logic(
    layer: LayerArchitecture,
) -> tuple[int, int]:
    arithmetic = layer.arithmetic
    dimensions = layer.dimensions

    activation_bits = int(
        arithmetic["activation_bits"]
    )
    weight_bits = int(
        arithmetic["weight_bits"]
    )
    accumulator_bits = int(
        arithmetic["accumulator_bits"]
    )
    kernel_elements = int(
        dimensions.get("kernel_elements", 1)
    )
    input_channels = int(
        dimensions.get("input_channels", 1)
    )

    explicit_logic = (
        layer.explicit_lanes
        * (
            45
            + 3 * activation_bits
            + 3 * weight_bits
            + 5 * accumulator_bits
        )
    )
    overlap_logic = (
        max(
            0,
            layer.effective_lanes
            - layer.explicit_lanes,
        )
        * (
            accumulator_bits
            + 20
        )
    )
    address_logic = (
        420
        + 40 * kernel_elements
        + 14 * input_channels
    )

    lut = (
        address_logic
        + explicit_logic
        + overlap_logic
    )
    ff = (
        520
        + address_logic
        + explicit_logic
        + 2 * overlap_logic
    )

    return lut, ff


def _pool_logic(
    layer: LayerArchitecture,
) -> tuple[int, int]:
    activation_bits = int(
        layer.arithmetic["activation_bits"]
    )
    accumulator_bits = int(
        layer.arithmetic["accumulator_bits"]
    )
    units = max(
        1,
        layer.effective_lanes,
    )

    if layer.op_type == "MaxPool":
        return (
            80 + units * (activation_bits + 12),
            90 + units * (activation_bits + 16),
        )

    return (
        100 + units * (accumulator_bits + 16),
        120 + units * (accumulator_bits + 20),
    )


def _elementwise_logic(
    layer: LayerArchitecture,
) -> tuple[int, int, int]:
    bits = int(
        layer.arithmetic["activation_bits"]
    )
    lanes = max(
        1,
        layer.explicit_lanes,
    )

    if layer.op_type == "Relu":
        return (
            35 + lanes * (bits + 8),
            40 + lanes * (bits + 10),
            0,
        )

    if layer.op_type == "LeakyRelu":
        dsp = lanes * _multiplier_dsp_cost(
            int(layer.arithmetic["multiply_left_bits"]),
            int(layer.arithmetic["multiply_right_bits"]),
            op_type=layer.op_type,
        )
        return (
            60 + lanes * (2 * bits + 18),
            70 + lanes * (2 * bits + 22),
            dsp,
        )

    if layer.op_type == "Add":
        return (
            45 + lanes * (bits + 12),
            55 + lanes * (bits + 14),
            0,
        )

    if layer.op_type == "Sigmoid":
        return (
            500 + lanes * (18 * bits + 160),
            560 + lanes * (20 * bits + 180),
            3 * lanes,
        )

    if layer.op_type == "Softmax":
        return (
            900 + 24 * bits,
            1000 + 26 * bits,
            2,
        )

    if layer.op_type in {"Flatten", "Reshape"}:
        return 20, 20, 0

    return (
        100 + lanes * (bits + 20),
        120 + lanes * (bits + 24),
        0,
    )


def estimate_architecture_layer_resources(
    layer: LayerArchitecture,
    raw_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if layer.op_type == "Dense":
        lut, ff = _dense_logic(layer)
    elif layer.op_type == "Conv":
        lut, ff = _conv_logic(layer)
    elif layer.op_type in {"MaxPool", "AvgPool"}:
        lut, ff = _pool_logic(layer)
    else:
        lut, ff, elementwise_dsp = (
            _elementwise_logic(layer)
        )

    multiplier_units = int(
        layer.arithmetic.get(
            "effective_multiplier_units",
            0,
        )
    )

    if layer.op_type in {"Dense", "Conv"}:
        dsp_per_multiplier = _multiplier_dsp_cost(
            int(
                layer.arithmetic[
                    "multiply_left_bits"
                ]
            ),
            int(
                layer.arithmetic[
                    "multiply_right_bits"
                ]
            ),
            op_type=layer.op_type,
        )
        dsp = (
            multiplier_units
            * dsp_per_multiplier
        )
    else:
        dsp_per_multiplier = 0
        dsp = elementwise_dsp

    weight_bram, bias_bram = _parameter_bram(
        layer,
        raw_cfg,
    )
    activation_bram = _activation_bram(
        layer,
        raw_cfg,
    )
    line_buffer_bram = _line_buffer_bram(
        layer,
    )

    bram18 = (
        weight_bram
        + bias_bram
        + activation_bram
        + line_buffer_bram
    )

    return {
        "layer_name": layer.name,
        "op_type": layer.op_type,
        "architecture": layer.to_dict(),
        "activation_bits": int(
            layer.arithmetic["activation_bits"]
        ),
        "weight_bits": int(
            layer.arithmetic["weight_bits"]
        ),
        "bias_bits": int(
            layer.arithmetic["bias_bits"]
        ),
        "accumulator_bits": int(
            layer.arithmetic["accumulator_bits"]
        ),
        "pe": int(
            layer.unroll.get(
                "out",
                layer.unroll.get("oc", 1),
            )
        ),
        "simd": int(
            layer.unroll.get(
                "in",
                layer.unroll.get("ic", 1),
            )
        ),
        "partition_factor": int(
            layer.memory["partition_factor"]
        ),
        "multiplier_lanes": multiplier_units,
        "multiplier_dsp_per_lane": (
            dsp_per_multiplier
        ),
        "dimensions": dict(layer.dimensions),
        "macs": int(
            layer.dimensions.get("macs", 0)
        ),
        "resource_components": {
            "arithmetic_lut": int(lut),
            "arithmetic_ff": int(ff),
            "arithmetic_dsp": int(dsp),
            "parameter_bram18": weight_bram,
            "bias_bram18": bias_bram,
            "activation_bram18": activation_bram,
            "line_buffer_bram18": line_buffer_bram,
        },
        "predicted_lut_raw": int(lut),
        "predicted_ff_raw": int(ff),
        "predicted_dsp_raw": int(dsp),
        "predicted_bram18_raw": int(bram18),
        "predicted_lut": int(lut),
        "predicted_ff": int(ff),
        "predicted_dsp": int(dsp),
        "predicted_bram18": int(bram18),
    }


def estimate_architecture_resources(
    architecture: HLSArchitecture,
    raw_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []

    for index, layer in enumerate(
        architecture.layers
    ):
        result = estimate_architecture_layer_resources(
            layer,
            raw_cfg,
        )
        result["layer_index"] = index
        results.append(result)

    return results