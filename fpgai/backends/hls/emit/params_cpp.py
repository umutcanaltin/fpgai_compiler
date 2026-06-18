from __future__ import annotations

from typing import List, Optional

import numpy as np


WEIGHT_ATTR_KEYS = (
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
)

BIAS_ATTR_KEYS = (
    "bias",
    "biases",
    "B",
    "b",
    "bias_data",
    "bias_values",
    "bias_name",
)


def _as_numeric_array(
    value,
) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        return None

    try:
        array = np.asarray(value)
    except Exception:
        return None

    if array.dtype.kind in {
        "U",
        "S",
        "O",
    }:
        return None

    return array


def _array_from_graph(
    graph,
    tensor_name: str | None,
) -> Optional[np.ndarray]:
    if not tensor_name:
        return None

    constants = getattr(
        graph,
        "constants",
        {},
    ) or {}

    if tensor_name in constants:
        array = _as_numeric_array(
            constants[tensor_name]
        )

        if array is not None:
            return array

    params = getattr(
        graph,
        "params",
        {},
    ) or {}

    if tensor_name in params:
        array = _as_numeric_array(
            params[tensor_name]
        )

        if array is not None:
            return array

    try:
        tensor = graph.get_tensor(
            tensor_name
        )
    except Exception:
        tensor = None

    if tensor is None:
        return None

    for attribute_name in (
        "data",
        "initializer",
        "value",
        "values",
    ):
        value = getattr(
            tensor,
            attribute_name,
            None,
        )
        array = _as_numeric_array(value)

        if array is not None:
            return array

    return None


def _array_from_attrs(
    graph,
    op,
    keys: tuple[str, ...],
) -> tuple[Optional[np.ndarray], Optional[str]]:
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
            array = _array_from_graph(
                graph,
                value,
            )

            if array is not None:
                return (
                    array,
                    f"op.attrs[{key!r}] -> "
                    f"graph tensor {value!r}",
                )

            continue

        array = _as_numeric_array(value)

        if array is not None:
            return (
                array,
                f"op.attrs[{key!r}]",
            )

    return None, None


def _normalize_length(
    array: np.ndarray,
    expected_count: int,
    *,
    parameter_name: str,
    op_name: str,
) -> np.ndarray:
    flattened = np.asarray(
        array
    ).reshape(-1)

    if expected_count <= 0:
        if flattened.size == 0:
            raise ValueError(
                f"{parameter_name} for op {op_name!r} "
                "is empty"
            )

        return flattened

    if flattened.size != expected_count:
        raise ValueError(
            f"{parameter_name} size mismatch for "
            f"op {op_name!r}: expected "
            f"{expected_count}, got {flattened.size}"
        )

    return flattened


def _format_scalar(
    value,
) -> str:
    numeric = float(value)

    if not np.isfinite(numeric):
        raise ValueError(
            "HLS embedded parameters cannot contain "
            f"non-finite value {numeric!r}"
        )

    # Decimal representation avoids NumPy-specific
    # formatting and remains valid C++.
    return f"{numeric:.17g}"


def _format_array(
    name: str,
    ctype: str,
    array: np.ndarray,
    *,
    const: bool = True,
    zero_init: bool = False,
) -> str:
    flattened = np.asarray(
        array
    ).reshape(-1)

    qualifier = "const " if const else ""

    if zero_init:
        return (
            f"{qualifier}{ctype} {name}"
            f"[{flattened.size}] = {{ 0 }};"
        )

    values = ", ".join(
        _format_scalar(value)
        for value in flattened
    )

    return (
        f"{qualifier}{ctype} {name}"
        f"[{flattened.size}] = {{ "
        f"{values} }};"
    )


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


def _dense_dimensions(
    op,
) -> tuple[int, int]:
    attrs = getattr(
        op,
        "attrs",
        {},
    ) or {}

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

    return input_features, output_features


def _resolve_dense_parameters(
    graph,
    op,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    str,
]:
    input_features, output_features = (
        _dense_dimensions(op)
    )

    weight_array = None
    bias_array = None
    weight_source = None
    bias_source = None

    if len(op.inputs) > 1:
        weight_name = op.inputs[1]
        weight_array = _array_from_graph(
            graph,
            weight_name,
        )

        if weight_array is not None:
            weight_source = (
                f"graph tensor {weight_name!r}"
            )

    if len(op.inputs) > 2:
        bias_name = op.inputs[2]
        bias_array = _array_from_graph(
            graph,
            bias_name,
        )

        if bias_array is not None:
            bias_source = (
                f"graph tensor {bias_name!r}"
            )

    if weight_array is None:
        weight_array, weight_source = (
            _array_from_attrs(
                graph,
                op,
                WEIGHT_ATTR_KEYS,
            )
        )

    if bias_array is None:
        bias_array, bias_source = (
            _array_from_attrs(
                graph,
                op,
                BIAS_ATTR_KEYS,
            )
        )

    if weight_array is None:
        raise ValueError(
            f"Dense weights not found for "
            f"op {op.name!r}"
        )

    weight_array = np.asarray(
        weight_array
    )

    if (
        input_features <= 0
        or output_features <= 0
    ):
        if weight_array.ndim != 2:
            raise ValueError(
                f"Dense op {op.name!r} requires "
                "in_features/out_features metadata "
                "or a two-dimensional weight array"
            )

        output_features = int(
            weight_array.shape[0]
        )
        input_features = int(
            weight_array.shape[1]
        )

    expected_weight_count = (
        input_features
        * output_features
    )

    weight_array = _normalize_length(
        weight_array,
        expected_weight_count,
        parameter_name="Dense weights",
        op_name=op.name,
    )

    if bias_array is None:
        bias_array = np.zeros(
            (output_features,),
            dtype=np.float32,
        )
        bias_source = "generated zero bias"

    bias_array = _normalize_length(
        bias_array,
        output_features,
        parameter_name="Dense bias",
        op_name=op.name,
    )

    return (
        weight_array,
        bias_array,
        weight_source or "unknown",
        bias_source or "unknown",
    )


def _resolve_conv_parameters(
    graph,
    op,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    str,
]:
    weight_array = None
    bias_array = None
    weight_source = None
    bias_source = None

    if len(op.inputs) > 1:
        weight_name = op.inputs[1]
        weight_array = _array_from_graph(
            graph,
            weight_name,
        )

        if weight_array is not None:
            weight_source = (
                f"graph tensor {weight_name!r}"
            )

    if len(op.inputs) > 2:
        bias_name = op.inputs[2]
        bias_array = _array_from_graph(
            graph,
            bias_name,
        )

        if bias_array is not None:
            bias_source = (
                f"graph tensor {bias_name!r}"
            )

    if weight_array is None:
        weight_array, weight_source = (
            _array_from_attrs(
                graph,
                op,
                WEIGHT_ATTR_KEYS,
            )
        )

    if bias_array is None:
        bias_array, bias_source = (
            _array_from_attrs(
                graph,
                op,
                BIAS_ATTR_KEYS,
            )
        )

    if weight_array is None:
        raise ValueError(
            f"Conv weights not found for "
            f"op {op.name!r}"
        )

    weight_array = np.asarray(
        weight_array
    )

    if weight_array.ndim != 4:
        raise ValueError(
            f"Conv weights for op {op.name!r} "
            "must have shape "
            "[out_channels, in_channels, kh, kw], "
            f"got {weight_array.shape}"
        )

    output_channels = int(
        weight_array.shape[0]
    )

    flattened_weights = (
        weight_array.reshape(-1)
    )

    if bias_array is None:
        bias_array = np.zeros(
            (output_channels,),
            dtype=np.float32,
        )
        bias_source = "generated zero bias"

    flattened_bias = _normalize_length(
        bias_array,
        output_channels,
        parameter_name="Conv bias",
        op_name=op.name,
    )

    return (
        flattened_weights,
        flattened_bias,
        weight_source or "unknown",
        bias_source or "unknown",
    )


def emit_params_cpp(
    graph,
    *,
    weights_mode: str = "embedded",
) -> str:
    normalized_mode = str(
        weights_mode
    ).strip().lower()

    lines: List[str] = [
        '#include "fpgai_params.h"',
        "",
        "namespace fpgai {",
        "",
    ]

    runtime_mode = normalized_mode in {
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
    }

    if normalized_mode != "embedded" and not runtime_mode:
        raise ValueError(
            f"Unsupported weights mode: "
            f"{weights_mode!r}"
        )

    parameter_index = 0

    for graph_index, op in enumerate(
        graph.ops
    ):
        if op.op_type not in {
            "Dense",
            "Conv",
        }:
            continue

        precision_tag = _precision_tag(
            op,
            graph_index,
        )
        weight_type = (
            f"{precision_tag}_wgt_t"
        )
        bias_type = (
            f"{precision_tag}_bias_t"
        )

        if op.op_type == "Dense":
            (
                weight_array,
                bias_array,
                weight_source,
                bias_source,
            ) = _resolve_dense_parameters(
                graph,
                op,
            )
        else:
            (
                weight_array,
                bias_array,
                weight_source,
                bias_source,
            ) = _resolve_conv_parameters(
                graph,
                op,
            )

        lines.append(
            f"// {op.op_type} layer "
            f"{op.name!r}"
        )
        lines.append(
            f"// Weight source: "
            f"{weight_source}"
        )
        lines.append(
            f"// Bias source: "
            f"{bias_source}"
        )
        lines.append(
            _format_array(
                f"W{parameter_index}",
                weight_type,
                weight_array,
                const=not runtime_mode,
                zero_init=runtime_mode,
            )
        )
        lines.append(
            _format_array(
                f"B{parameter_index}",
                bias_type,
                bias_array,
                const=not runtime_mode,
                zero_init=runtime_mode,
            )
        )
        lines.append("")

        parameter_index += 1

    lines.extend(
        [
            "} // namespace fpgai",
            "",
        ]
    )

    return "\n".join(lines)

# Sprint 11D source fix: runtime modes require actual W*/B* definitions,
# otherwise Vitis C-sim links deeplearn.cpp against fpgai::W0/B0/... and
# fails with undefined references.  Embedded mode remains the original
# const initialized arrays; runtime modes allocate mutable arrays initialized
# to zero and populated by the generated stream/DDR loaders.
_fpgai_runtime_defs_previous_emit_params_cpp = emit_params_cpp


def emit_params_cpp(graph, *, weights_mode: str = "embedded") -> str:
    normalized_mode = str(weights_mode).strip().lower()
    if normalized_mode == "embedded":
        return _fpgai_runtime_defs_previous_emit_params_cpp(graph, weights_mode=weights_mode)
    if normalized_mode not in {"stream", "streamed", "ddr", "dma_ddr"}:
        raise ValueError(f"Unsupported weights mode: {weights_mode!r}")

    lines: List[str] = [
        '#include "fpgai_params.h"',
        "",
        "namespace fpgai {",
        "",
        "// Runtime weight storage definitions.",
        "// Populated by deeplearn.cpp from stream/DDR interfaces.",
    ]

    parameter_index = 0
    for graph_index, op in enumerate(graph.ops):
        if op.op_type not in {"Dense", "Conv"}:
            continue
        precision_tag = _precision_tag(op, graph_index)
        weight_type = f"{precision_tag}_wgt_t"
        bias_type = f"{precision_tag}_bias_t"
        if op.op_type == "Dense":
            weight_array, bias_array, weight_source, bias_source = _resolve_dense_parameters(graph, op)
        else:
            weight_array, bias_array, weight_source, bias_source = _resolve_conv_parameters(graph, op)
        weight_count = int(np.asarray(weight_array).reshape(-1).size)
        bias_count = int(np.asarray(bias_array).reshape(-1).size)
        lines.append(f"// {op.op_type} layer {op.name!r}")
        lines.append(f"// Runtime source mode: {normalized_mode}")
        lines.append(f"// Original weight source: {weight_source}")
        lines.append(f"// Original bias source: {bias_source}")
        lines.append(f"{weight_type} W{parameter_index}[{weight_count}] = {{}};")
        lines.append(f"{bias_type} B{parameter_index}[{bias_count}] = {{}};")
        lines.append("")
        parameter_index += 1

    lines.extend(["} // namespace fpgai", ""])
    return "\n".join(lines)
