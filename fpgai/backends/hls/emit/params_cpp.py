from __future__ import annotations

from typing import List, Optional

import numpy as np

from fpgai.backends.hls.emit.params_h import _conv_sizes, _dense_sizes


def _normalise_storage_impl(storage_impl: str | None) -> str:
    value = str(storage_impl or "bram").strip().lower()
    aliases = {
        "embedded": "bram",
        "on_chip": "bram",
        "onchip": "bram",
        "block": "bram",
        "block_ram": "bram",
        "bram": "bram",
        "uram": "uram",
        "ultra": "uram",
        "ultra_ram": "uram",
        "lutram": "lutram",
        "lut_ram": "lutram",
        "distributed": "lutram",
        "ddr": "ddr",
        "external": "ddr",
        "external_ddr": "ddr",
        "dma_ddr": "ddr",
        "stream": "stream",
        "streaming": "stream",
    }
    return aliases.get(value, "bram")


def _bind_storage_impl(storage_impl: str | None) -> str | None:
    value = _normalise_storage_impl(storage_impl)
    if value == "bram":
        return "bram"
    if value == "uram":
        return "uram"
    if value == "lutram":
        return "lutram"
    return None


def _parameter_binding_pragmas(graph, storage_impl: str | None) -> list[str]:
    impl = _bind_storage_impl(storage_impl)
    if impl is None:
        return [
            "",
            f"// FPGAI storage binding: {storage_impl or 'none'} uses runtime/external storage; no local BIND_STORAGE pragmas emitted.",
        ]

    lines = [
        "",
        f"// FPGAI storage binding: parameter arrays requested for {impl.upper()}.",
        "// FPGAI note: file-scope BIND_STORAGE pragmas are disabled because Vitis HLS csynth rejects them on global const arrays.",
    ]

    parameter_index = 0
    for op in getattr(graph, "ops", []):
        op_type = str(getattr(op, "op_type", "")).lower()
        if op_type not in {"dense", "gemm", "conv"}:
            continue

        if op_type in {"dense", "gemm"}:
            weight_count, bias_count = _dense_sizes(graph, op)
        else:
            weight_count, bias_count = _conv_sizes(graph, op)

        if weight_count > 0:
            lines.append(
                f"// FPGAI storage binding: {impl} requested for W{parameter_index}; file-scope BIND_STORAGE disabled."
            )
        if bias_count > 0:
            lines.append(
                f"// FPGAI storage binding: {impl} requested for B{parameter_index}; file-scope BIND_STORAGE disabled."
            )

        parameter_index += 1

    return lines




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
) -> str:
    flattened = np.asarray(
        array
    ).reshape(-1)

    values = ", ".join(
        _format_scalar(value)
        for value in flattened
    )

    return (
        f"const {ctype} {name}"
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
    storage_impl: str | None = "bram",
) -> str:
    normalized_mode = str(weights_mode).strip().lower()

    if normalized_mode not in {
        "embedded",
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
    }:
        raise ValueError(
            f"Unsupported weights mode: {weights_mode!r}"
        )

    runtime_parameters = normalized_mode in {
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
    }

    lines: List[str] = [
        '#include "fpgai_params.h"',
        "",
        "namespace fpgai {",
        "",
    ]

    if runtime_parameters:
        lines.append(
            "// Mutable runtime parameter storage. Arrays are initialized from the "
            "ONNX parameters so C-simulation and correctness benchmarking use "
            "the same weights as the Python reference. The top function can "
            "still overwrite them through AXI stream or DDR preload modes."
        )
        lines.append("")

        preload_entries: list[tuple[str, str, int]] = []
        parameter_index = 0
        total_runtime_words = 0

        for graph_index, op in enumerate(graph.ops):
            if op.op_type not in {"Dense", "Conv"}:
                continue

            precision_tag = _precision_tag(op, graph_index)
            weight_type = f"{precision_tag}_wgt_t"
            bias_type = f"{precision_tag}_bias_t"

            if op.op_type == "Dense":
                (
                    weight_array,
                    bias_array,
                    weight_source,
                    bias_source,
                ) = _resolve_dense_parameters(graph, op)
            else:
                (
                    weight_array,
                    bias_array,
                    weight_source,
                    bias_source,
                ) = _resolve_conv_parameters(graph, op)

            weight_array = np.asarray(weight_array).reshape(-1)
            bias_array = np.asarray(bias_array).reshape(-1)

            lines.append(f"// {op.op_type} layer {op.name!r}")
            lines.append(f"// Weight source: {weight_source}")
            lines.append(f"// Bias source: {bias_source}")
            # Runtime arrays must be mutable and intentionally uninitialized.
            # Vitis HLS rejects initialized global/static arrays when they are
            # bound to URAM (HLS 214-221).  Keep the reference ONNX values in
            # separate const *_init arrays used only by the testbench/preload
            # helpers; the top function writes runtime data into W*/B* through
            # AXI stream or DDR before inference.
            lines.append(
                f"{weight_type} W{parameter_index}[{int(weight_array.size)}];"
            )
            lines.append(
                f"{bias_type} B{parameter_index}[{int(bias_array.size)}];"
            )
            lines.append(
                _format_array(
                    f"W{parameter_index}_init",
                    weight_type,
                    weight_array,
                )
            )
            lines.append(
                _format_array(
                    f"B{parameter_index}_init",
                    bias_type,
                    bias_array,
                )
            )
            lines.append("")

            preload_entries.append((f"W{parameter_index}_init", weight_type, int(weight_array.size)))
            preload_entries.append((f"B{parameter_index}_init", bias_type, int(bias_array.size)))
            total_runtime_words += int(weight_array.size) + int(bias_array.size)
            parameter_index += 1

        lines.extend(
            [
                "static unsigned int fpgai_value_to_bits_float(float value) {",
                "    union { float f; unsigned int i; } u;",
                "    u.f = value;",
                "    return u.i;",
                "}",
                "",
                "template <typename T>",
                "static void fpgai_push_value(hls::stream<fpgai_axis_t>& weight_stream, T value, bool last) {",
                "    fpgai_axis_t packet;",
                "    packet.data = fpgai_value_to_bits_float((float)value);",
                "    packet.keep = -1;",
                "    packet.strb = -1;",
                "    packet.last = last ? 1 : 0;",
                "    weight_stream.write(packet);",
                "}",
                "",
                "template <typename T>",
                "static ap_uint<32> fpgai_pack_value(T value) {",
                "    return ap_uint<32>(fpgai_value_to_bits_float((float)value));",
                "}",
                "",
                "int fpgai_runtime_weight_word_count() {",
                f"    return {total_runtime_words};",
                "}",
                "",
                "void fpgai_preload_runtime_weights(hls::stream<fpgai_axis_t>& weight_stream) {",
                "    int emitted = 0;",
            ]
        )

        for symbol, _ctype, count in preload_entries:
            lines.append(f"    for (int i = 0; i < {count}; ++i) {{")
            lines.append(f"        const bool last = (emitted == {total_runtime_words - 1});")
            lines.append(f"        fpgai_push_value(weight_stream, {symbol}[i], last);")
            lines.append("        ++emitted;")
            lines.append("    }")

        lines.extend(
            [
                "}",
                "",
                "void fpgai_fill_runtime_weight_words(ap_uint<32>* weights_mem, int max_words) {",
                "    int emitted = 0;",
            ]
        )

        for symbol, _ctype, count in preload_entries:
            lines.append(f"    for (int i = 0; i < {count} && emitted < max_words; ++i) {{")
            lines.append(f"        weights_mem[emitted++] = fpgai_pack_value({symbol}[i]);")
            lines.append("    }")

        lines.extend(
            [
                "}",
                "",
                "} // namespace fpgai",
                "",
            ]
        )
        return "\n".join(lines)

    parameter_index = 0

    for graph_index, op in enumerate(graph.ops):
        if op.op_type not in {"Dense", "Conv"}:
            continue

        precision_tag = _precision_tag(op, graph_index)
        weight_type = f"{precision_tag}_wgt_t"
        bias_type = f"{precision_tag}_bias_t"

        if op.op_type == "Dense":
            (
                weight_array,
                bias_array,
                weight_source,
                bias_source,
            ) = _resolve_dense_parameters(graph, op)
        else:
            (
                weight_array,
                bias_array,
                weight_source,
                bias_source,
            ) = _resolve_conv_parameters(graph, op)

        lines.append(f"// {op.op_type} layer {op.name!r}")
        lines.append(f"// Weight source: {weight_source}")
        lines.append(f"// Bias source: {bias_source}")
        lines.append(_format_array(f"W{parameter_index}", weight_type, weight_array))
        lines.append(_format_array(f"B{parameter_index}", bias_type, bias_array))
        lines.append("")

        parameter_index += 1

    lines.extend(_parameter_binding_pragmas(graph, storage_impl))
    lines.extend(["} // namespace fpgai", ""])
    return "\n".join(lines)
