from __future__ import annotations
from fpgai.backends.hls.emit.architecture_comments import emit_layer_architecture_comments
from fpgai.backends.hls.emit.dense_tiling_codegen import apply_dense_tiling_to_top_source
from fpgai.backends.hls.emit.conv_tiling_codegen import apply_conv_tiling_to_top_source

from typing import Any, Dict, Tuple

import numpy as np

from fpgai.ir.graph import Graph


def _object_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(value, dict):
        return value

    return {}


def _plan_map(
    compile_plan: Any,
) -> Dict[str, Dict[str, Any]]:
    if compile_plan is None:
        return {}

    plans = getattr(
        compile_plan,
        "layer_plans",
        None,
    )

    if plans is None and isinstance(
        compile_plan,
        dict,
    ):
        plans = compile_plan.get(
            "layer_plans",
            [],
        )

    result = {}

    for plan in plans or []:
        data = _object_dict(plan)
        name = data.get("node_name")

        if name:
            result[name] = data

    return result


def _architecture_section(
    layer_plan: Dict[str, Any],
    section: str,
) -> Dict[str, Any]:
    architecture = layer_plan.get(
        "architecture",
        {},
    )
    if not isinstance(architecture, dict):
        return {}
    value = architecture.get(section, {})
    return value if isinstance(value, dict) else {}


def _positive_codegen_int(
    value: Any,
    default: int = 1,
) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _layer_codegen_values(
    layer_plan: Dict[str, Any],
    *,
    op_type: str,
) -> Dict[str, int]:
    pipeline = _architecture_section(
        layer_plan,
        "pipeline",
    )
    parallelism = _architecture_section(
        layer_plan,
        "parallelism",
    )
    partitioning = _architecture_section(
        layer_plan,
        "partitioning",
    )

    unroll = parallelism.get(
        "unroll",
        layer_plan.get("unroll", {}),
    )
    if not isinstance(unroll, dict):
        unroll = {}

    targets = partitioning.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}

    notes = layer_plan.get("notes", {})
    if not isinstance(notes, dict):
        notes = {}

    factor = _positive_codegen_int(
        partitioning.get(
            "factor",
            notes.get("partition_factor", 1),
        )
    )

    if op_type == "Dense":
        output_unroll = _positive_codegen_int(
            unroll.get("out", 1)
        )
        input_unroll = _positive_codegen_int(
            unroll.get("in", 1)
        )
    else:
        output_unroll = _positive_codegen_int(
            unroll.get("oc", 1)
        )
        input_unroll = _positive_codegen_int(
            unroll.get("ic", 1)
        )

    return {
        "pipeline_ii": _positive_codegen_int(
            pipeline.get(
                "ii",
                layer_plan.get("pipeline_ii", 1),
            )
        ),
        "input_unroll": input_unroll,
        "output_unroll": output_unroll,
        "input_partition": _positive_codegen_int(
            targets.get("input", factor)
        ),
        "output_partition": _positive_codegen_int(
            targets.get("output", factor)
        ),
        "weight_partition": _positive_codegen_int(
            targets.get("weight", factor)
        ),
    }




def _emit_parallel_artifact_comment(
    lines: list[str],
    indent: str,
    *,
    op_type: str,
    layer_name: str,
    codegen: Dict[str, int],
) -> None:
    lines.append(
        f"{indent}// FPGAI parallel artifact: "
        f"op={op_type} "
        f"name={layer_name} "
        f"pipeline_ii={codegen['pipeline_ii']} "
        f"input_unroll={codegen['input_unroll']} "
        f"output_unroll={codegen['output_unroll']} "
        f"input_partition={codegen['input_partition']} "
        f"output_partition={codegen['output_partition']} "
        f"weight_partition={codegen['weight_partition']}"
    )


def _comm_map(
    communication_plan: Any,
) -> Dict[str, Dict[str, Any]]:
    if communication_plan is None:
        return {}

    edges = getattr(
        communication_plan,
        "edges",
        None,
    )

    if edges is None and isinstance(
        communication_plan,
        dict,
    ):
        edges = communication_plan.get(
            "edges",
            [],
        )

    result = {}

    for edge in edges or []:
        data = _object_dict(edge)
        name = data.get("tensor_name")

        if name:
            result[name] = data

    return result


def _memory_map(
    memory_plan: Any,
) -> Dict[str, Dict[str, Any]]:
    if memory_plan is None:
        return {}

    placements = getattr(
        memory_plan,
        "placements",
        None,
    )

    if placements is None and isinstance(
        memory_plan,
        dict,
    ):
        placements = memory_plan.get(
            "placements",
            [],
        )

    result = {}

    for placement in placements or []:
        data = _object_dict(placement)
        name = data.get("tensor_name")

        if name:
            result[name] = data

    return result


def _strip_batch(
    shape,
) -> Tuple[int, ...]:
    if shape is None:
        return tuple()

    result = tuple(
        int(value)
        for value in shape
    )

    if (
        len(result) > 1
        and result[0] == 1
    ):
        return result[1:]

    return result


def _flat_size(
    shape,
) -> int:
    if not shape:
        return 1

    return int(np.prod(shape))


def _as_chw(
    shape: Tuple[int, ...],
) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(
            f"Expected CHW shape, got {shape}"
        )

    return (
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
    )


def _conv_out_dim(
    input_dimension: int,
    kernel: int,
    stride: int,
    total_padding: int,
    dilation: int = 1,
) -> int:
    return (
        (
            input_dimension
            + total_padding
            - dilation * (kernel - 1)
            - 1
        )
        // stride
    ) + 1


def _infer_out_shape(
    graph: Graph,
    op,
    current_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    output_name = op.outputs[0]
    output_spec = graph.get_tensor(
        output_name
    )

    if (
        output_spec is not None
        and getattr(
            output_spec,
            "shape",
            None,
        )
    ):
        return _strip_batch(
            output_spec.shape
        )

    if op.op_type == "Conv":
        _, input_height, input_width = (
            _as_chw(current_shape)
        )

        output_channels = None
        kernel_height = 3
        kernel_width = 3

        if len(op.inputs) > 1:
            weight_name = op.inputs[1]
            weight_spec = graph.get_tensor(
                weight_name
            )

            if (
                weight_spec is not None
                and getattr(
                    weight_spec,
                    "shape",
                    None,
                )
            ):
                weight_shape = tuple(
                    int(value)
                    for value in weight_spec.shape
                )
            elif (
                hasattr(graph, "constants")
                and weight_name in graph.constants
            ):
                weight_shape = tuple(
                    int(value)
                    for value in graph.constants[
                        weight_name
                    ].shape
                )
            else:
                raise ValueError(
                    f"Conv weights not found for "
                    f"op {op.name!r}"
                )

            if len(weight_shape) != 4:
                raise ValueError(
                    f"Conv weights must be rank four, "
                    f"got {weight_shape}"
                )

            output_channels = weight_shape[0]
            kernel_height = weight_shape[2]
            kernel_width = weight_shape[3]

        if output_channels is None:
            raise ValueError(
                f"Cannot resolve output channels "
                f"for Conv op {op.name!r}"
            )

        strides = op.attrs.get(
            "strides",
            [1, 1],
        )
        pads = op.attrs.get(
            "pads",
            [0, 0, 0, 0],
        )
        dilations = op.attrs.get(
            "dilations",
            [1, 1],
        )

        stride_height = int(strides[0])
        stride_width = int(strides[1])
        pad_top, pad_left, pad_bottom, pad_right = [
            int(value)
            for value in pads
        ]
        dilation_height = int(dilations[0])
        dilation_width = int(dilations[1])

        output_height = _conv_out_dim(
            input_height,
            kernel_height,
            stride_height,
            pad_top + pad_bottom,
            dilation_height,
        )
        output_width = _conv_out_dim(
            input_width,
            kernel_width,
            stride_width,
            pad_left + pad_right,
            dilation_width,
        )

        return (
            int(output_channels),
            output_height,
            output_width,
        )

    if op.op_type in {
        "MaxPool",
        "AvgPool",
    }:
        channels, input_height, input_width = (
            _as_chw(current_shape)
        )

        kernel = op.attrs.get(
            "kernel_shape",
            [2, 2],
        )
        strides = op.attrs.get(
            "strides",
            [2, 2],
        )
        pads = op.attrs.get(
            "pads",
            [0, 0, 0, 0],
        )

        kernel_height = int(kernel[0])
        kernel_width = int(kernel[1])
        stride_height = int(strides[0])
        stride_width = int(strides[1])
        pad_top, pad_left, pad_bottom, pad_right = [
            int(value)
            for value in pads
        ]

        output_height = _conv_out_dim(
            input_height,
            kernel_height,
            stride_height,
            pad_top + pad_bottom,
        )
        output_width = _conv_out_dim(
            input_width,
            kernel_width,
            stride_width,
            pad_left + pad_right,
        )

        return (
            channels,
            output_height,
            output_width,
        )

    if op.op_type in {
        "Relu",
        "Sigmoid",
        "LeakyRelu",
        "Add",
        "BatchNormalization",
        "Softmax",
    }:
        return tuple(current_shape)

    if op.op_type in {
        "Flatten",
        "Reshape",
    }:
        return (
            _flat_size(current_shape),
        )

    if op.op_type == "Dense":
        output_features = op.attrs.get(
            "out_features",
            op.attrs.get("out"),
        )

        if (
            output_features is None
            and len(op.inputs) > 1
        ):
            weight_name = op.inputs[1]
            weight_spec = graph.get_tensor(
                weight_name
            )

            if (
                weight_spec is not None
                and getattr(
                    weight_spec,
                    "shape",
                    None,
                )
            ):
                output_features = int(
                    weight_spec.shape[0]
                )
            elif (
                hasattr(graph, "constants")
                and weight_name in graph.constants
            ):
                output_features = int(
                    graph.constants[
                        weight_name
                    ].shape[0]
                )

        if output_features is None:
            raise ValueError(
                f"Cannot resolve Dense output size "
                f"for op {op.name!r}"
            )

        return (
            int(output_features),
        )

    raise ValueError(
        f"Cannot infer output shape for "
        f"{op.name} ({op.op_type})"
    )


def _placement_comment(
    memory_info: Dict[str, Any] | None,
) -> str:
    if not memory_info:
        return "unknown"

    return (
        f"{memory_info.get('region', 'unknown')} / "
        f"size={memory_info.get('size_bytes', 'unknown')} "
        "bytes"
    )


def _communication_comment(
    communication_info: Dict[str, Any] | None,
) -> str:
    if not communication_info:
        return "unknown"

    return (
        f"{communication_info.get('direction', 'unknown')} / "
        f"{communication_info.get('encoding', 'raw')}"
    )


def _emit_plan_comments(
    lines: list[str],
    indent: str,
    layer_plan: Dict[str, Any],
    output_memory: Dict[str, Any] | None,
    output_communication: Dict[str, Any] | None,
) -> None:
    lines.append(
        f"{indent}// precision_mode: "
        f"{layer_plan.get('precision_mode')}"
    )
    lines.append(
        f"{indent}// activation bits: "
        f"{layer_plan.get('act_bits')}"
    )
    lines.append(
        f"{indent}// weight bits: "
        f"{layer_plan.get('weight_bits')}"
    )
    lines.append(
        f"{indent}// tile: "
        f"{layer_plan.get('tile', {})}"
    )
    lines.append(
        f"{indent}// unroll: "
        f"{layer_plan.get('unroll', {})}"
    )
    lines.append(
        f"{indent}// output placement: "
        f"{_placement_comment(output_memory)}"
    )
    lines.append(
        f"{indent}// output communication: "
        f"{_communication_comment(output_communication)}"
    )


def _emit_storage_pragma(
    lines: list[str],
    variable_name: str,
    memory_info: Dict[str, Any] | None,
) -> None:
    region = str(
        (memory_info or {}).get(
            "region",
            "BRAM",
        )
    ).upper()

    implementation = (
        "uram"
        if region == "URAM"
        else "bram"
    )

    lines.append(
        f"#pragma HLS BIND_STORAGE "
        f"variable={variable_name} "
        f"type=ram_1p impl={implementation}"
    )


def _weight_memory_info_from_layer_plan(
    layer_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Return HLS storage preference for a layer's runtime/local weight cache.

    The planner stores this under architecture.memory.weight_region in recent
    FPGAI plans.  Older plans may expose a top-level memory dictionary.  This
    helper deliberately falls back to BRAM so existing designs remain stable.
    """

    memory = _architecture_section(
        layer_plan,
        "memory",
    )
    if not memory:
        candidate = layer_plan.get(
            "memory",
            {},
        )
        memory = candidate if isinstance(candidate, dict) else {}

    region = str(
        memory.get(
            "weight_region",
            memory.get(
                "region",
                "BRAM",
            ),
        )
    ).upper()

    # External-DDR weights are loaded into a local cache before compute; the
    # cache itself must still be implemented on-chip.  Treat DDR as BRAM here.
    if region not in {"BRAM", "URAM"}:
        region = "BRAM"

    return {
        "region": region,
        "size_bytes": "weight_cache",
    }


def _emit_weight_storage_pragmas(
    lines: list[str],
    parameter_index: int,
    layer_plan: Dict[str, Any],
) -> None:
    memory_info = _weight_memory_info_from_layer_plan(
        layer_plan,
    )
    lines.append(
        f"    // Weight cache placement: "
        f"{_placement_comment(memory_info)}"
    )
    _emit_storage_pragma(
        lines,
        f"W{parameter_index}",
        memory_info,
    )
    _emit_storage_pragma(
        lines,
        f"B{parameter_index}",
        memory_info,
    )


def _precision_tag(
    op,
    index: int,
) -> str:
    return str(
        op.attrs.get(
            "precision_tag",
            f"op{index}",
        )
    )


def _activation_type(
    op,
    index: int,
) -> str:
    return (
        f"{_precision_tag(op, index)}_act_t"
    )


def _weight_type(
    op,
    index: int,
) -> str:
    return (
        f"{_precision_tag(op, index)}_wgt_t"
    )


def _bias_type(
    op,
    index: int,
) -> str:
    return (
        f"{_precision_tag(op, index)}_bias_t"
    )


def _accumulator_type(
    op,
    index: int,
) -> str:
    return (
        f"{_precision_tag(op, index)}_acc_t"
    )


def emit_top_cpp(
    graph: Graph,
    *,
    top_name: str,
    weights_mode: str,
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
) -> str:
    if not graph.inputs:
        raise ValueError(
            "Inference graph has no inputs"
        )

    if not graph.outputs:
        raise ValueError(
            "Inference graph has no outputs"
        )

    if not graph.ops:
        raise ValueError(
            "Inference graph has no operations"
        )

    normalized_weights_mode = str(
        weights_mode
    ).strip().lower()

    if normalized_weights_mode != "embedded":
        raise ValueError(
            "Mixed-precision inference currently "
            "requires weights mode 'embedded'"
        )

    plan_by_name = _plan_map(
        compile_plan
    )
    communication_by_tensor = _comm_map(
        communication_plan
    )
    memory_by_tensor = _memory_map(
        memory_plan
    )

    input_spec = graph.get_tensor(
        graph.inputs[0]
    )
    input_shape = (
        tuple(input_spec.shape)
        if input_spec is not None
        and input_spec.shape
        else (1,)
    )
    current_shape = _strip_batch(
        input_shape
    )

    first_type = _activation_type(
        graph.ops[0],
        0,
    )

    lines: list[str] = [
        "#include <hls_stream.h>",
        "#include <ap_axi_sdata.h>",
        '#include "fpgai_types.h"',
        '#include "fpgai_params.h"',
        '#include "layers/dense.h"',
        '#include "layers/conv.h"',
        '#include "layers/pool.h"',
        '#include "layers/activations.h"',
        '#include "layers/batchnorm.h"',
        "#if defined(FPGAI_DEBUG_DUMP) && "
        "!defined(__SYNTHESIS__)",
        "#include <fstream>",
        "#endif",
        "",
        "typedef ap_axis<32, 0, 0, 0> axis_t;",
        "using namespace fpgai;",
        "",
        "template<typename T>",
        "static inline T bits_to_value(unsigned int bits) {",
        "    union { unsigned int i; float f; } converter;",
        "    converter.i = bits;",
        "    return (T)converter.f;",
        "}",
        "",
        "template<typename T>",
        "static inline unsigned int value_to_bits(T value) {",
        "    union { unsigned int i; float f; } converter;",
        "    converter.f = (float)value;",
        "    return converter.i;",
        "}",
        "",
        "#if defined(FPGAI_DEBUG_DUMP) && "
        "!defined(__SYNTHESIS__)",
        "template<typename T>",
        "static inline void fpgai_dump_tensor(",
        "    const char* path,",
        "    const T* data,",
        "    int count",
        ") {",
        "    std::ofstream output(path, std::ios::binary);",
        "    for (int index = 0; index < count; ++index) {",
        "        float value = (float)data[index];",
        "        output.write(",
        "            reinterpret_cast<const char*>(&value),",
        "            sizeof(float)",
        "        );",
        "    }",
        "}",
        "#endif",
        "",
        f'extern "C" void {top_name}(',
        "    hls::stream<axis_t>& in_stream,",
        "    hls::stream<axis_t>& out_stream",
        ") {",
        "#pragma HLS INTERFACE axis port=in_stream",
        "#pragma HLS INTERFACE axis port=out_stream",
        "#pragma HLS INTERFACE s_axilite "
        "port=return bundle=control",
        "",
    ]

    input_name = graph.inputs[0]
    input_size = _flat_size(
        current_shape
    )
    current_buffer = "layer_in"
    current_type = first_type

    lines.append(
        f"    // Input transfer: "
        f"{_communication_comment(communication_by_tensor.get(input_name))}"
    )
    lines.append(
        f"    // Input placement: "
        f"{_placement_comment(memory_by_tensor.get(input_name))}"
    )
    lines.append(
        f"    {current_type} "
        f"{current_buffer}[{input_size}];"
    )
    _emit_storage_pragma(
        lines,
        current_buffer,
        memory_by_tensor.get(input_name),
    )
    lines.extend(
        [
            (
                f"    for (int index = 0; "
                f"index < {input_size}; ++index) {{"
            ),
            "        axis_t packet = in_stream.read();",
            (
                f"        {current_buffer}[index] = "
                f"bits_to_value<{current_type}>("
                "packet.data.to_uint());"
            ),
            "    }",
            "",
        ]
    )

    parameter_index = 0
    batchnorm_index = 0

    for layer_index, op in enumerate(
        graph.ops
    ):
        output_name = op.outputs[0]
        output_shape = _infer_out_shape(
            graph,
            op,
            current_shape,
        )
        output_size = _flat_size(
            output_shape
        )
        output_buffer = (
            f"layer_{layer_index}_out"
        )
        output_type = _activation_type(
            op,
            layer_index,
        )
        weight_type = _weight_type(
            op,
            layer_index,
        )
        bias_type = _bias_type(
            op,
            layer_index,
        )
        accumulator_type = _accumulator_type(
            op,
            layer_index,
        )

        layer_plan = plan_by_name.get(
            op.name,
            {},
        )
        output_memory = memory_by_tensor.get(
            output_name
        )
        output_communication = (
            communication_by_tensor.get(
                output_name
            )
        )

        lines.append(
            f"    // Layer {layer_index}: "
            f"{op.op_type} ({op.name})"
        )
        lines.append(
            f"    // Input type: {current_type}"
        )
        lines.append(
            f"    // Output type: {output_type}"
        )
        lines.append(
            f"    // Accumulator type: "
            f"{accumulator_type}"
        )
        _emit_plan_comments(
            lines,
            "    ",
            layer_plan,
            output_memory,
            output_communication,
        )
        lines.append(
            f"    {output_type} "
            f"{output_buffer}[{output_size}];"
        )
        _emit_storage_pragma(
            lines,
            output_buffer,
            output_memory,
        )

        if op.op_type == "Conv":
            codegen = _layer_codegen_values(
                layer_plan,
                op_type="Conv",
            )
            _emit_parallel_artifact_comment(
                lines,
                "    ",
                op_type="Conv",
                layer_name=str(op.name),
                codegen=codegen,
            )
            architecture_arguments = (
                f", {codegen['pipeline_ii']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}"
                if layer_plan
                else ""
            )
            (
                input_channels,
                input_height,
                input_width,
            ) = _as_chw(current_shape)
            (
                output_channels,
                output_height,
                output_width,
            ) = _as_chw(output_shape)

            if len(op.inputs) < 2:
                raise ValueError(
                    f"Conv op {op.name!r} has no weights"
                )

            weight_name = op.inputs[1]

            if (
                hasattr(graph, "constants")
                and weight_name in graph.constants
            ):
                weight_shape = tuple(
                    int(value)
                    for value in graph.constants[
                        weight_name
                    ].shape
                )
            else:
                weight_spec = graph.get_tensor(
                    weight_name
                )

                if (
                    weight_spec is None
                    or not weight_spec.shape
                ):
                    raise ValueError(
                        f"Conv weights not found for "
                        f"op {op.name!r}"
                    )

                weight_shape = tuple(
                    int(value)
                    for value in weight_spec.shape
                )

            kernel_height = int(
                weight_shape[2]
            )
            kernel_width = int(
                weight_shape[3]
            )

            if kernel_height != kernel_width:
                raise ValueError(
                    "HLS Conv currently requires "
                    "square kernels"
                )

            strides = op.attrs.get(
                "strides",
                [1, 1],
            )
            pads = op.attrs.get(
                "pads",
                [0, 0, 0, 0],
            )

            if int(strides[0]) != int(strides[1]):
                raise ValueError(
                    "HLS Conv currently requires "
                    "equal height/width strides"
                )

            if not (
                int(pads[0])
                == int(pads[1])
                == int(pads[2])
                == int(pads[3])
            ):
                raise ValueError(
                    "HLS Conv currently requires "
                    "symmetric padding"
                )

            _emit_weight_storage_pragmas(
                lines,
                parameter_index,
                layer_plan,
            )
            lines.append(
                "    conv2d<"
                f"{input_height}, "
                f"{input_width}, "
                f"{input_channels}, "
                f"{output_height}, "
                f"{output_width}, "
                f"{output_channels}, "
                f"{kernel_height}, "
                f"{int(strides[0])}, "
                f"{int(pads[0])}, "
                f"{current_type}, "
                f"{output_type}, "
                f"{weight_type}, "
                f"{bias_type}, "
                f"{accumulator_type}"
                f"{architecture_arguments}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}, "
                f"W{parameter_index}, "
                f"B{parameter_index}"
                ");"
            )
            parameter_index += 1

        elif op.op_type == "Dense":
            codegen = _layer_codegen_values(
                layer_plan,
                op_type="Dense",
            )
            _emit_parallel_artifact_comment(
                lines,
                "    ",
                op_type="Dense",
                layer_name=str(op.name),
                codegen=codegen,
            )
            architecture_arguments = (
                f", {codegen['pipeline_ii']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}"
                if layer_plan
                else ""
            )
            input_features = _flat_size(
                current_shape
            )
            output_features = _flat_size(
                output_shape
            )

            _emit_weight_storage_pragmas(
                lines,
                parameter_index,
                layer_plan,
            )
            lines.append(
                "    dense_out_in<"
                f"{input_features}, "
                f"{output_features}, "
                f"{current_type}, "
                f"{output_type}, "
                f"{weight_type}, "
                f"{bias_type}, "
                f"{accumulator_type}"
                f"{architecture_arguments}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}, "
                f"W{parameter_index}, "
                f"B{parameter_index}"
                ");"
            )
            parameter_index += 1

        elif op.op_type == "Relu":
            lines.append(
                "    relu_typed<"
                f"{output_size}, "
                f"{current_type}, "
                f"{output_type}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}"
                ");"
            )

        elif op.op_type == "LeakyRelu":
            alpha = float(
                op.attrs.get(
                    "alpha",
                    0.1,
                )
            )
            lines.append(
                "    leaky_relu_typed<"
                f"{output_size}, "
                f"{current_type}, "
                f"{output_type}, "
                f"{accumulator_type}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}, "
                f"({accumulator_type})"
                f"{alpha:.17g}"
                ");"
            )

        elif op.op_type == "Sigmoid":
            lines.append(
                "    sigmoid_typed<"
                f"{output_size}, "
                f"{current_type}, "
                f"{output_type}, "
                f"{accumulator_type}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}"
                ");"
            )

        elif op.op_type == "Softmax":
            lines.append(
                "    softmax_typed<"
                f"{output_size}, "
                f"{current_type}, "
                f"{output_type}, "
                f"{accumulator_type}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}"
                ");"
            )

        elif op.op_type in {
            "MaxPool",
            "AvgPool",
        }:
            (
                input_channels,
                input_height,
                input_width,
            ) = _as_chw(current_shape)
            (
                _,
                output_height,
                output_width,
            ) = _as_chw(output_shape)

            kernel = op.attrs.get(
                "kernel_shape",
                [2, 2],
            )
            strides = op.attrs.get(
                "strides",
                [2, 2],
            )
            pads = op.attrs.get(
                "pads",
                [0, 0, 0, 0],
            )

            if int(kernel[0]) != int(kernel[1]):
                raise ValueError(
                    "HLS pooling currently requires "
                    "square kernels"
                )

            if int(strides[0]) != int(strides[1]):
                raise ValueError(
                    "HLS pooling currently requires "
                    "equal height/width strides"
                )

            if any(int(value) != 0 for value in pads):
                raise ValueError(
                    "HLS pooling currently does not "
                    "support padding"
                )

            if op.op_type == "MaxPool":
                function_name = (
                    "maxpool2d_typed"
                )
                template_types = (
                    f"{current_type}, "
                    f"{output_type}"
                )
            else:
                function_name = (
                    "avgpool2d_typed"
                )
                template_types = (
                    f"{current_type}, "
                    f"{output_type}, "
                    f"{accumulator_type}"
                )

            lines.append(
                f"    {function_name}<"
                f"{input_height}, "
                f"{input_width}, "
                f"{input_channels}, "
                f"{int(kernel[0])}, "
                f"{int(strides[0])}, "
                f"{output_height}, "
                f"{output_width}, "
                f"{template_types}"
                ">("
                f"{current_buffer}, "
                f"{output_buffer}"
                ");"
            )

        elif op.op_type in {
            "Flatten",
            "Reshape",
        }:
            if (
                op.op_type == "Reshape"
                and len(current_shape) == 3
                and len(output_shape) == 1
            ):
                (
                    channels,
                    height,
                    width,
                ) = _as_chw(current_shape)

                lines.extend(
                    [
                        (
                            f"    for (int channel = 0; "
                            f"channel < {channels}; "
                            "++channel) {"
                        ),
                        (
                            f"        for (int row = 0; "
                            f"row < {height}; ++row) {{"
                        ),
                        (
                            f"            for (int column = 0; "
                            f"column < {width}; ++column) {{"
                        ),
                        (
                            f"                const int source = "
                            f"(row * {width} + column) "
                            f"* {channels} + channel;"
                        ),
                        (
                            f"                const int destination = "
                            f"(channel * {height} + row) "
                            f"* {width} + column;"
                        ),
                        (
                            f"                {output_buffer}"
                            f"[destination] = "
                            f"({output_type})"
                            f"{current_buffer}[source];"
                        ),
                        "            }",
                        "        }",
                        "    }",
                    ]
                )
            else:
                lines.append(
                    "    reshape_copy_typed<"
                    f"{output_size}, "
                    f"{current_type}, "
                    f"{output_type}"
                    ">("
                    f"{current_buffer}, "
                    f"{output_buffer}"
                    ");"
                )

        elif op.op_type == "Add":
            raise RuntimeError(
                "General graph Add requires tensor "
                "liveness/branch resolution and is not "
                "yet supported by mixed-precision top emission"
            )

        elif op.op_type == "BatchNormalization":
            raise RuntimeError(
                "BatchNormalization mixed-precision "
                "inference requires typed BN parameters "
                "and will be implemented separately"
            )

        else:
            raise RuntimeError(
                f"Unsupported mixed-precision HLS op: "
                f"{op.op_type}"
            )

        lines.extend(
            [
                "#if defined(FPGAI_DEBUG_DUMP) && "
                "!defined(__SYNTHESIS__)",
                (
                    f'    fpgai_dump_tensor('
                    f'"{op.name}.bin", '
                    f"{output_buffer}, "
                    f"{output_size}"
                    ");"
                ),
                "#endif",
                "",
            ]
        )

        current_buffer = output_buffer
        current_type = output_type
        current_shape = output_shape

    output_name = graph.outputs[0]
    output_size = _flat_size(
        current_shape
    )

    lines.append(
        f"    // Output transfer: "
        f"{_communication_comment(communication_by_tensor.get(output_name))}"
    )
    lines.append(
        f"    // Output placement: "
        f"{_placement_comment(memory_by_tensor.get(output_name))}"
    )
    lines.extend(
        [
            (
                f"    for (int index = 0; "
                f"index < {output_size}; ++index) {{"
            ),
            "        axis_t packet;",
            (
                f"        packet.data = "
                f"value_to_bits<{current_type}>("
                f"{current_buffer}[index]);"
            ),
            "        packet.keep = -1;",
            "        packet.strb = -1;",
            (
                f"        packet.last = "
                f"(index == {output_size - 1}) "
                "? 1 : 0;"
            ),
            "        out_stream.write(packet);",
            "    }",
            "}",
            "",
        ]
    )

    return "\n".join(lines)

# FPGAI architecture-comment wrapper.
# This keeps the original emitter implementation intact while making
# generated HLS sources self-describing for experiment artifacts.
_fpgai_original_emit_top_cpp = emit_top_cpp


def emit_top_cpp(*args, **kwargs):
    source = _fpgai_original_emit_top_cpp(*args, **kwargs)

    compile_plan = kwargs.get("compile_plan")
    if compile_plan is None:
        try:
            import inspect

            bound = inspect.signature(_fpgai_original_emit_top_cpp).bind_partial(
                *args,
                **kwargs,
            )
            compile_plan = bound.arguments.get("compile_plan")
        except Exception:
            compile_plan = None

    comments = emit_layer_architecture_comments(compile_plan)
    return comments + source

# FPGAI dense-tiling wrapper.
# This wraps the architecture-comment emitter and then rewrites Dense calls
# whose compile plan contains tile sizes into tiled Dense kernels.
_fpgai_dense_tiling_original_emit_top_cpp = emit_top_cpp


def emit_top_cpp(*args, **kwargs):
    source = _fpgai_dense_tiling_original_emit_top_cpp(*args, **kwargs)

    graph = kwargs.get("graph")
    if graph is None and args:
        graph = args[0]

    compile_plan = kwargs.get("compile_plan")

    return apply_dense_tiling_to_top_source(
        source,
        graph,
        compile_plan,
    )

# FPGAI convolution-tiling wrapper.
# This runs after the dense-tiling wrapper and rewrites Conv calls whose
# compile plan contains channel or spatial tile sizes.
_fpgai_conv_tiling_original_emit_top_cpp = emit_top_cpp


def emit_top_cpp(*args, **kwargs):
    source = _fpgai_conv_tiling_original_emit_top_cpp(*args, **kwargs)

    graph = kwargs.get("graph")
    if graph is None and args:
        graph = args[0]

    compile_plan = kwargs.get("compile_plan")

    return apply_conv_tiling_to_top_source(
        source,
        graph,
        compile_plan,
    )

# FPGAI runtime-weight wrapper.
# The core emitter still generates the normal AXI input/output datapath.
# For non-embedded modes we reuse that generated body and add real runtime
# parameter interfaces before inference starts.
_fpgai_runtime_weight_previous_emit_top_cpp = emit_top_cpp


def _fpgai_runtime_weight_mode(weights_mode: str) -> str:
    return str(weights_mode).strip().lower()


def _fpgai_runtime_weight_specs(graph):
    from fpgai.backends.hls.emit.params_h import _conv_sizes, _dense_sizes

    specs = []
    parameter_index = 0
    for graph_index, op in enumerate(graph.ops):
        if op.op_type not in {"Conv", "Dense"}:
            continue

        precision_tag = _precision_tag(op, graph_index)
        if op.op_type == "Conv":
            weight_count, bias_count = _conv_sizes(graph, op)
        else:
            weight_count, bias_count = _dense_sizes(graph, op)

        if weight_count <= 0:
            raise ValueError(
                f"{op.op_type} weights could not be resolved for op {op.name!r}"
            )
        if bias_count <= 0:
            raise ValueError(
                f"{op.op_type} bias size could not be resolved for op {op.name!r}"
            )

        specs.append(
            (
                parameter_index,
                precision_tag,
                int(weight_count),
                int(bias_count),
                op.op_type,
                op.name,
            )
        )
        parameter_index += 1

    return specs


def _fpgai_runtime_recover_graph_top(args, kwargs):
    graph = kwargs.get("graph")
    top_name = kwargs.get("top_name")
    if graph is None and args:
        graph = args[0]
    if top_name is None:
        try:
            import inspect

            bound = inspect.signature(
                _fpgai_runtime_weight_previous_emit_top_cpp
            ).bind_partial(*args, **kwargs)
            graph = graph or bound.arguments.get("graph")
            top_name = top_name or bound.arguments.get("top_name")
        except Exception:
            pass
    return graph, top_name


def _fpgai_insert_runtime_helpers(source: str) -> str:
    if "fpgai_load_stream_vector" in source:
        return source

    helper = """
template<typename T, int N>
static void fpgai_load_stream_vector(hls::stream<axis_t>& weight_stream, T out[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        axis_t packet = weight_stream.read();
        out[i] = bits_to_value<T>(packet.data.to_uint());
    }
}

template<typename T, int N>
static void fpgai_load_ddr_vector(const ap_uint<32>* weights_mem, int& offset, T out[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = bits_to_value<T>(weights_mem[offset + i].to_uint());
    }
    offset += N;
}
"""

    # The helper uses both axis_t and bits_to_value<T>().  Insert it only
    # after the generated typedef and scalar conversion helpers.  Inserting
    # before them compiles as hls::stream<int&> and causes Vitis CSim errors.
    value_to_bits_marker = "template<typename T>\nstatic inline unsigned int value_to_bits(T value)"
    value_to_bits_pos = source.find(value_to_bits_marker)
    if value_to_bits_pos >= 0:
        block_end = source.find("\n}\n", value_to_bits_pos)
        if block_end >= 0:
            insert_at = block_end + len("\n}\n")
            return source[:insert_at] + "\n" + helper + "\n" + source[insert_at:]

    # Fallback: insert before the top function, but only if the normal helper
    # marker was not found.  This path is kept for older generated source
    # layouts.
    signature = 'extern "C" void '
    position = source.find(signature)
    if position >= 0:
        return source[:position] + helper + "\n" + source[position:]
    return source + "\n" + helper


def _fpgai_rewrite_runtime_signature(source: str, *, top_name: str, mode: str) -> str:
    if "#include <ap_int.h>" not in source:
        source = source.replace(
            "#include <ap_axi_sdata.h>",
            "#include <ap_axi_sdata.h>\n#include <ap_int.h>",
        )

    original_signature = (
        f'extern "C" void {top_name}(\n'
        "    hls::stream<axis_t>& in_stream,\n"
        "    hls::stream<axis_t>& out_stream\n"
        ") {"
    )

    if mode in {"stream", "streamed"}:
        runtime_signature = (
            f'extern "C" void {top_name}(\n'
            "    hls::stream<axis_t>& in_stream,\n"
            "    hls::stream<axis_t>& out_stream,\n"
            "    hls::stream<axis_t>& weight_stream,\n"
            "    int mode\n"
            ") {"
        )
    else:
        runtime_signature = (
            f'extern "C" void {top_name}(\n'
            "    hls::stream<axis_t>& in_stream,\n"
            "    hls::stream<axis_t>& out_stream,\n"
            "    const ap_uint<32>* weights_mem\n"
            ") {"
        )

    if original_signature not in source:
        raise ValueError("Could not rewrite top signature for runtime weight mode")
    return source.replace(original_signature, runtime_signature, 1)


def _fpgai_runtime_load_block(graph, *, mode: str) -> str:
    specs = _fpgai_runtime_weight_specs(graph)
    lines = []
    if mode in {"stream", "streamed"}:
        lines.extend(
            [
                "    // Runtime weight preload mode.",
                "    // mode == 0 loads W*/B* from weight_stream and returns.",
                "    if (mode == 0) {",
            ]
        )
        for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
            lines.append(f"        // {op_type} {op_name}: W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}]")
            lines.append(
                f"        fpgai_load_stream_vector<{precision_tag}_wgt_t, {weight_count}>(weight_stream, W{parameter_index});"
            )
            lines.append(
                f"        fpgai_load_stream_vector<{precision_tag}_bias_t, {bias_count}>(weight_stream, B{parameter_index});"
            )
        lines.extend(
            [
                "        return;",
                "    }",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "    // Runtime DDR weight load.",
                "    int fpgai_weight_offset = 0;",
            ]
        )
        for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
            lines.append(f"    // {op_type} {op_name}: W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}]")
            lines.append(
                f"    fpgai_load_ddr_vector<{precision_tag}_wgt_t, {weight_count}>(weights_mem, fpgai_weight_offset, W{parameter_index});"
            )
            lines.append(
                f"    fpgai_load_ddr_vector<{precision_tag}_bias_t, {bias_count}>(weights_mem, fpgai_weight_offset, B{parameter_index});"
            )
        lines.append("")
    return "\n".join(lines)


def _fpgai_insert_runtime_load_block(source: str, graph, *, mode: str) -> str:
    block = _fpgai_runtime_load_block(graph, mode=mode)
    if mode in {"stream", "streamed"}:
        pragma_marker = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
        replacement = (
            "#pragma HLS INTERFACE axis port=weight_stream\n"
            "#pragma HLS INTERFACE s_axilite port=mode bundle=control\n"
            "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n"
            + block
            + "\n"
        )
    else:
        pragma_marker = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
        replacement = (
            "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights\n"
            "#pragma HLS INTERFACE s_axilite port=weights_mem bundle=control\n"
            "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n"
            + block
            + "\n"
        )

    if pragma_marker not in source:
        raise ValueError("Could not insert runtime weight HLS pragmas")
    return source.replace(pragma_marker, replacement, 1)


def emit_top_cpp(*args, **kwargs):
    requested_mode = _fpgai_runtime_weight_mode(kwargs.get("weights_mode", "embedded"))
    if requested_mode not in {"stream", "streamed", "ddr", "dma_ddr"}:
        return _fpgai_runtime_weight_previous_emit_top_cpp(*args, **kwargs)

    graph, top_name = _fpgai_runtime_recover_graph_top(args, kwargs)
    if graph is None or top_name is None:
        raise ValueError("Runtime weight top emission requires graph and top_name")

    updateed_kwargs = dict(kwargs)
    updateed_kwargs["weights_mode"] = "embedded"
    source = _fpgai_runtime_weight_previous_emit_top_cpp(*args, **updateed_kwargs)
    source = _fpgai_insert_runtime_helpers(source)
    source = _fpgai_rewrite_runtime_signature(source, top_name=top_name, mode=requested_mode)
    source = _fpgai_insert_runtime_load_block(source, graph, mode=requested_mode)

    planning_comment = (
        f"// Requested weights mode: {requested_mode}\n"
        "// Non-embedded weight modes are represented in the memory/compile plan; "
        "generated standalone HLS C simulation keeps embedded constants for reproducibility.\n"
    )
    return planning_comment + source
