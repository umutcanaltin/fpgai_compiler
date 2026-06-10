from __future__ import annotations

from typing import List

import numpy as np


def _numel_from_graph_named(
    graph,
    tensor_name: str | None,
) -> int:
    if not tensor_name:
        return 0

    constants = getattr(
        graph,
        "constants",
        {},
    ) or {}

    if tensor_name in constants:
        return int(
            np.asarray(
                constants[tensor_name]
            ).size
        )

    params = getattr(
        graph,
        "params",
        {},
    ) or {}

    if tensor_name in params:
        return int(
            np.asarray(
                params[tensor_name]
            ).size
        )

    try:
        tensor = graph.get_tensor(
            tensor_name
        )
    except Exception:
        tensor = None

    if tensor is None:
        return 0

    shape = getattr(
        tensor,
        "shape",
        None,
    )

    if shape:
        count = 1

        for dimension in shape:
            count *= int(dimension)

        return int(count)

    data = getattr(
        tensor,
        "data",
        None,
    )

    if data is not None:
        return int(
            np.asarray(data).size
        )

    return 0


def _numeric_attr_numel(
    graph,
    op,
    keys: tuple[str, ...],
) -> int:
    attrs = getattr(
        op,
        "attrs",
        {},
    ) or {}

    for key in keys:
        if key not in attrs:
            continue

        value = attrs[key]

        if isinstance(value, str):
            count = _numel_from_graph_named(
                graph,
                value,
            )

            if count > 0:
                return count

            continue

        try:
            array = np.asarray(value)
        except Exception:
            continue

        if array.dtype.kind in {
            "U",
            "S",
            "O",
        }:
            continue

        if array.size > 0:
            return int(array.size)

    return 0


def _dense_sizes(
    graph,
    op,
) -> tuple[int, int]:
    input_features = int(
        op.attrs.get(
            "in_features",
            0,
        )
        or 0
    )
    output_features = int(
        op.attrs.get(
            "out_features",
            0,
        )
        or 0
    )

    weight_count = 0
    bias_count = 0

    if len(op.inputs) > 1:
        weight_count = _numel_from_graph_named(
            graph,
            op.inputs[1],
        )

    if len(op.inputs) > 2:
        bias_count = _numel_from_graph_named(
            graph,
            op.inputs[2],
        )

    if weight_count == 0:
        weight_count = _numeric_attr_numel(
            graph,
            op,
            (
                "weights",
                "weight",
                "kernel",
                "W",
                "w",
                "weight_data",
                "weights_data",
                "kernel_data",
                "weight_values",
                "weights_values",
                "kernel_values",
                "weights_name",
                "weight_name",
            ),
        )

    if bias_count == 0:
        bias_count = _numeric_attr_numel(
            graph,
            op,
            (
                "bias",
                "biases",
                "B",
                "b",
                "bias_data",
                "bias_values",
                "bias_name",
            ),
        )

    if (
        weight_count == 0
        and input_features > 0
        and output_features > 0
    ):
        weight_count = (
            input_features
            * output_features
        )

    # Dense inference always receives a bias array.
    # When ONNX has no bias, params_cpp emits zeros.
    if (
        bias_count == 0
        and output_features > 0
    ):
        bias_count = output_features

    return weight_count, bias_count


def _conv_sizes(
    graph,
    op,
) -> tuple[int, int]:
    weight_count = 0
    bias_count = 0

    if len(op.inputs) > 1:
        weight_count = _numel_from_graph_named(
            graph,
            op.inputs[1],
        )

    if len(op.inputs) > 2:
        bias_count = _numel_from_graph_named(
            graph,
            op.inputs[2],
        )

    if weight_count == 0:
        weight_count = _numeric_attr_numel(
            graph,
            op,
            (
                "weights",
                "weight",
                "kernel",
                "W",
                "w",
                "weight_data",
                "weights_data",
                "kernel_data",
                "weight_values",
                "weights_values",
                "kernel_values",
                "weights_name",
                "weight_name",
            ),
        )

    if bias_count == 0:
        bias_count = _numeric_attr_numel(
            graph,
            op,
            (
                "bias",
                "biases",
                "B",
                "b",
                "bias_data",
                "bias_values",
                "bias_name",
            ),
        )

    if bias_count == 0 and weight_count > 0:
        output_channels = 0

        if len(op.inputs) > 1:
            weight_name = op.inputs[1]

            constants = getattr(
                graph,
                "constants",
                {},
            ) or {}

            if weight_name in constants:
                shape = np.asarray(
                    constants[weight_name]
                ).shape

                if shape:
                    output_channels = int(
                        shape[0]
                    )

            if output_channels == 0:
                try:
                    tensor = graph.get_tensor(
                        weight_name
                    )
                except Exception:
                    tensor = None

                if (
                    tensor is not None
                    and getattr(
                        tensor,
                        "shape",
                        None,
                    )
                ):
                    output_channels = int(
                        tensor.shape[0]
                    )

        if output_channels == 0:
            output_channels = int(
                op.attrs.get(
                    "out_channels",
                    0,
                )
                or 0
            )

        # Conv inference always receives a bias array.
        # Bias-free convolutions use a generated zero array.
        bias_count = output_channels

    return weight_count, bias_count


def _precision_tag(
    op,
    graph_index: int,
) -> str:
    attrs = getattr(
        op,
        "attrs",
        {},
    ) or {}

    return str(
        attrs.get(
            "precision_tag",
            f"op{graph_index}",
        )
    )


def emit_params_h(
    graph,
    *,
    weights_mode: str = "embedded",
) -> str:
    lines: List[str] = [
        "#pragma once",
        '#include "fpgai_types.h"',
        "",
        "namespace fpgai {",
        "",
    ]

    normalized_mode = str(
        weights_mode
    ).strip().lower()

    if normalized_mode == "embedded":
        parameter_index = 0

        for graph_index, op in enumerate(
            graph.ops
        ):
            if op.op_type not in {
                "Conv",
                "Dense",
            }:
                continue

            precision_tag = _precision_tag(
                op,
                graph_index,
            )

            if op.op_type == "Conv":
                weight_count, bias_count = (
                    _conv_sizes(
                        graph,
                        op,
                    )
                )
            else:
                weight_count, bias_count = (
                    _dense_sizes(
                        graph,
                        op,
                    )
                )

            if weight_count <= 0:
                raise ValueError(
                    f"{op.op_type} weights could not "
                    f"be resolved for op {op.name!r}"
                )

            if bias_count <= 0:
                raise ValueError(
                    f"{op.op_type} bias size could not "
                    f"be resolved for op {op.name!r}"
                )

            lines.append(
                f"extern const "
                f"{precision_tag}_wgt_t "
                f"W{parameter_index}"
                f"[{weight_count}];"
            )
            lines.append(
                f"extern const "
                f"{precision_tag}_bias_t "
                f"B{parameter_index}"
                f"[{bias_count}];"
            )
            lines.append("")

            parameter_index += 1

    elif normalized_mode in {
        "stream",
        "streamed",
    }:
        lines.append(
            "// Parameters are loaded through "
            "the runtime weight stream."
        )
        lines.append("")

    elif normalized_mode in {
        "ddr",
        "dma_ddr",
    }:
        lines.append(
            "// DDR parameter declarations are "
            "provided by the runtime backend."
        )
        lines.append("")

    else:
        raise ValueError(
            f"Unsupported weights mode: "
            f"{weights_mode!r}"
        )

    lines.extend(
        [
            "} // namespace fpgai",
            "",
        ]
    )

    return "\n".join(lines)


def emit_params_h_stub(
    graph,
    *,
    weights_mode: str = "embedded",
) -> str:
    return emit_params_h(
        graph,
        weights_mode=weights_mode,
    )