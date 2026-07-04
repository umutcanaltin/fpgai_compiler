from __future__ import annotations

from fpgai.numerics.precision_policy import default_precision_policy
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
    precision_policy: Dict[str, Any] | None = None,
) -> None:
    activation_bits = layer_plan.get("act_bits")
    weight_bits = layer_plan.get("weight_bits")
    if precision_policy:
        activation_bits = (
            precision_policy.get("activation", {}) or {}
        ).get("total_bits", activation_bits)
        weight_bits = (
            precision_policy.get("weight", {}) or {}
        ).get("total_bits", weight_bits)

    lines.append(
        f"{indent}// precision_mode: "
        f"{layer_plan.get('precision_mode')}"
    )
    lines.append(
        f"{indent}// activation bits: "
        f"{activation_bits}"
    )
    lines.append(
        f"{indent}// weight bits: "
        f"{weight_bits}"
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
    lines.append(
        "    // FPGAI storage binding requested for embedded parameter "
        f"W{parameter_index}/B{parameter_index}; "
        "top-level BIND_STORAGE is disabled for initialized W/B arrays "
        "because Vitis HLS rejects initialized global/static parameter "
        "arrays bound this way. Use runtime-loaded URAM buffers for real "
        "URAM parameter storage."
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
    raw_cfg: Any = None,
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
        "static const int FPGAI_AXIS_WORD_BITS = 32;",
        "",
        "template<typename T, int VALUE_BITS>",
        "static inline T fpgai_unpack_axis_value(axis_t packet, int lane) {",
        "    ap_uint<VALUE_BITS> raw = packet.data.range(",
        "        ((lane + 1) * VALUE_BITS) - 1,",
        "        lane * VALUE_BITS",
        "    );",
        "    T value;",
        "    value.range(VALUE_BITS - 1, 0) = raw;",
        "    return value;",
        "}",
        "",
        "template<typename T, int VALUE_BITS>",
        "static inline void fpgai_pack_axis_value(axis_t& packet, T value, int lane) {",
        "    ap_uint<VALUE_BITS> raw = value.range(VALUE_BITS - 1, 0);",
        "    packet.data.range(",
        "        ((lane + 1) * VALUE_BITS) - 1,",
        "        lane * VALUE_BITS",
        "    ) = raw;",
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
    precision_policy = default_precision_policy(raw_cfg or {})
    act_precision = precision_policy.get("activation", {})
    act_bits = int(act_precision.get("total_bits", 16))
    act_per_axis = max(1, 32 // act_bits)

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
            f"    static const int FPGAI_ACT_BITS = {act_bits};",
            f"    static const int FPGAI_ACT_PER_AXIS = {act_per_axis};",
            (
                f"    for (int base = 0; "
                f"base < {input_size}; base += FPGAI_ACT_PER_AXIS) {{"
            ),
            "#pragma HLS PIPELINE II=1",
            "        axis_t packet = in_stream.read();",
            "        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {",
            "#pragma HLS UNROLL",
            "            int index = base + lane;",
            f"            if (index < {input_size}) {{",
            (
                f"                {current_buffer}[index] = "
                f"fpgai_unpack_axis_value<{current_type}, FPGAI_ACT_BITS>("
                "packet, lane);"
            ),
            "            }",
            "        }",
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
            precision_policy,
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
                f"    for (int base = 0; "
                f"base < {output_size}; base += FPGAI_ACT_PER_AXIS) {{"
            ),
            "#pragma HLS PIPELINE II=1",
            "        axis_t packet;",
            "        packet.data = 0;",
            "        packet.keep = -1;",
            "        packet.strb = -1;",
            "        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {",
            "#pragma HLS UNROLL",
            "            int index = base + lane;",
            f"            if (index < {output_size}) {{",
            (
                f"                fpgai_pack_axis_value<{current_type}, FPGAI_ACT_BITS>("
                f"packet, {current_buffer}[index], lane);"
            ),
            "            }",
            "        }",
            (
                f"        packet.last = "
                f"(base + FPGAI_ACT_PER_AXIS >= {output_size}) "
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


def _fpgai_graph_has_conv_weights(graph) -> bool:
    return any(getattr(op, "op_type", None) == "Conv" for op in getattr(graph, "ops", []) or [])


def _fpgai_split_cpp_args(arg_text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in arg_text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args




def _fpgai_ddr_weight_offsets_by_type(graph, op_type: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    offset = 0
    for _parameter_index, _precision_tag, weight_count, bias_count, spec_op_type, _op_name in _fpgai_runtime_weight_specs(graph):
        weight_base = offset
        bias_base = offset + weight_count
        if spec_op_type == op_type:
            offsets.append((weight_base, bias_base))
        offset += weight_count + bias_count
    return offsets

def _fpgai_ddr_tiled_helper_cpp() -> str:
    return """

// FPGAI real DDR-tiled Dense helper.
// Full weights live in weights_mem. Only input_tile, weight_tile, and acc_tile
// are allocated on-chip during compute; no full local W/B replica is created.
#ifndef FPGAI_DDR_BITS_HELPER
#define FPGAI_DDR_BITS_HELPER
template<typename T>
static inline T fpgai_ddr_bits_to_value(unsigned int bits) {
    union { unsigned int i; float f; } converter;
    converter.i = bits;
    return (T)converter.f;
}
#endif

template<
    int IN,
    int OUT,
    int TILE_IN,
    int TILE_OUT,
    typename IN_T,
    typename OUT_T,
    typename W_T,
    typename B_T,
    typename ACC_T,
    int PIPELINE_II = 1,
    int IN_UNROLL = 1,
    int OUT_UNROLL = 1,
    int IN_PARTITION = 1,
    int OUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void dense_out_in_ddr_tiled(
    const IN_T input[IN],
    OUT_T output[OUT],
    const ap_uint<32>* weights_mem,
    int weight_base,
    int bias_base
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1

    for (int out_base = 0; out_base < OUT; out_base += TILE_OUT) {
        ACC_T acc_tile[TILE_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1

        dense_ddr_init_output_tile:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            acc_tile[out_inner] =
                (out_idx < OUT)
                ? (ACC_T)fpgai_ddr_bits_to_value<B_T>(weights_mem[bias_base + out_idx].to_uint())
                : (ACC_T)0;
        }

        for (int in_base = 0; in_base < IN; in_base += TILE_IN) {
            IN_T input_tile[TILE_IN];
            W_T weight_tile[TILE_OUT][TILE_IN];
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2

            dense_ddr_load_input_tile:
            for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                const int in_idx = in_base + in_inner;
                input_tile[in_inner] = (in_idx < IN) ? input[in_idx] : (IN_T)0;
            }

            dense_ddr_load_weight_tile_out:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_ddr_load_weight_tile_in:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    const int in_idx = in_base + in_inner;
                    weight_tile[out_inner][in_inner] =
                        (out_idx < OUT && in_idx < IN)
                        ? fpgai_ddr_bits_to_value<W_T>(weights_mem[weight_base + out_idx * IN + in_idx].to_uint())
                        : (W_T)0;
                }
            }

            dense_ddr_compute_tile_out:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_ddr_compute_tile_in:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    if (out_idx < OUT) {
                        acc_tile[out_inner] +=
                            (ACC_T)input_tile[in_inner] *
                            (ACC_T)weight_tile[out_inner][in_inner];
                    }
                }
            }
        }

        dense_ddr_store_output_tile:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            if (out_idx < OUT) {
                output[out_idx] = (OUT_T)acc_tile[out_inner];
            }
        }
    }
}
"""


def _fpgai_rewrite_dense_calls_for_ddr_tiled(source: str, graph) -> str:
    import re
    call_re = re.compile(
        r"dense_out_in(?P<tiled>_tiled)?\s*<(?P<template>[^>]*)>\s*\((?P<args>[^;]*)\)\s*;",
        re.MULTILINE | re.DOTALL,
    )
    offsets = _fpgai_ddr_weight_offsets_by_type(graph, "Dense")

    call_index = 0
    used = False

    def replace(match: re.Match[str]) -> str:
        nonlocal call_index, used
        args = _fpgai_split_cpp_args(match.group("args"))
        if len(args) < 2:
            return match.group(0)
        template_parts = [part.strip() for part in match.group("template").split(",")]
        if len(template_parts) < 2:
            return match.group(0)
        if match.group("tiled"):
            new_template = ", ".join(template_parts)
        else:
            in_count = template_parts[0]
            out_count = template_parts[1]
            new_template = ", ".join([in_count, out_count, in_count, out_count, *template_parts[2:]])
        weight_base, bias_base = offsets[call_index] if call_index < len(offsets) else (0, 0)
        call_index += 1
        used = True
        return (
            f"dense_out_in_ddr_tiled<{new_template}>("
            f"{args[0]}, {args[1]}, weights_mem, {weight_base}, {bias_base});"
        )

    rewritten = call_re.sub(replace, source)
    if used and "FPGAI real DDR-tiled Dense helper" not in rewritten:
        if "#include <ap_int.h>" not in rewritten:
            rewritten = rewritten.replace("#include <ap_axi_sdata.h>", "#include <ap_axi_sdata.h>\n#include <ap_int.h>")
        signature = 'extern "C" void '
        pos = rewritten.find(signature)
        if pos >= 0:
            rewritten = rewritten[:pos] + _fpgai_ddr_tiled_helper_cpp() + "\n" + rewritten[pos:]
        else:
            rewritten = _fpgai_ddr_tiled_helper_cpp() + "\n" + rewritten
    return rewritten




def _fpgai_conv_ddr_tiled_helper_cpp() -> str:
    return r'''

// FPGAI real DDR-tiled Conv helper.
// Full Conv weights live in weights_mem. Only input_tile, conv_weight_tile,
// and acc_tile are allocated on-chip during compute; no full local Conv W/B
// replica is created.
#ifndef FPGAI_DDR_BITS_HELPER
#define FPGAI_DDR_BITS_HELPER
template<typename T>
static inline T fpgai_ddr_bits_to_value(unsigned int bits) {
    union { unsigned int i; float f; } converter;
    converter.i = bits;
    return (T)converter.f;
}
#endif

template<
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD,
    int TILE_OC,
    int TILE_OH,
    int TILE_OW,
    int TILE_IC,
    typename IN_T,
    typename OUT_T,
    typename W_T,
    typename B_T,
    typename ACC_T,
    int PIPELINE_II = 1,
    int OC_UNROLL = 1,
    int IC_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void conv2d_ddr_tiled(
    const IN_T input[IN_H * IN_W * IN_C],
    OUT_T output[OUT_H * OUT_W * OUT_C],
    const ap_uint<32>* weights_mem,
    int weight_base,
    int bias_base
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION dim=1

    for (int oc_base = 0; oc_base < OUT_C; oc_base += TILE_OC) {
        for (int oh_base = 0; oh_base < OUT_H; oh_base += TILE_OH) {
            for (int ow_base = 0; ow_base < OUT_W; ow_base += TILE_OW) {
                ACC_T acc_tile[TILE_OC][TILE_OH][TILE_OW];
                IN_T input_tile[TILE_IC][TILE_OH * STRIDE + K][TILE_OW * STRIDE + K];
                W_T conv_weight_tile[TILE_OC][TILE_IC][K][K];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv_weight_tile complete dim=2

                conv_ddr_init_acc_oc:
                for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                    const int oc = oc_base + oc_inner;
                    conv_ddr_init_acc_oh:
                    for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                        conv_ddr_init_acc_ow:
                        for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
                            const int oh = oh_base + oh_inner;
                            const int ow = ow_base + ow_inner;
                            acc_tile[oc_inner][oh_inner][ow_inner] =
                                (oc < OUT_C && oh < OUT_H && ow < OUT_W)
                                ? (ACC_T)fpgai_ddr_bits_to_value<B_T>(weights_mem[bias_base + oc].to_uint())
                                : (ACC_T)0;
                        }
                    }
                }

                for (int ic_base = 0; ic_base < IN_C; ic_base += TILE_IC) {
                    conv_ddr_load_input_ic:
                    for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS UNROLL factor=IC_UNROLL
                        const int ic = ic_base + ic_inner;
                        conv_ddr_load_input_h:
                        for (int tile_ih = 0; tile_ih < TILE_OH * STRIDE + K; ++tile_ih) {
                            conv_ddr_load_input_w:
                            for (int tile_iw = 0; tile_iw < TILE_OW * STRIDE + K; ++tile_iw) {
#pragma HLS PIPELINE II=PIPELINE_II
                                const int ih = oh_base * STRIDE + tile_ih - PAD;
                                const int iw = ow_base * STRIDE + tile_iw - PAD;
                                input_tile[ic_inner][tile_ih][tile_iw] =
                                    (ic < IN_C && ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                                    ? input[(ic * IN_H + ih) * IN_W + iw]
                                    : (IN_T)0;
                            }
                        }
                    }

                    conv_ddr_load_weight_oc:
                    for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                        const int oc = oc_base + oc_inner;
                        conv_ddr_load_weight_ic:
                        for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS UNROLL factor=IC_UNROLL
                            const int ic = ic_base + ic_inner;
                            conv_ddr_load_weight_kh:
                            for (int kh = 0; kh < K; ++kh) {
                                conv_ddr_load_weight_kw:
                                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=PIPELINE_II
                                    const int weight_index = ((oc * IN_C + ic) * K + kh) * K + kw;
                                    conv_weight_tile[oc_inner][ic_inner][kh][kw] =
                                        (oc < OUT_C && ic < IN_C)
                                        ? fpgai_ddr_bits_to_value<W_T>(weights_mem[weight_base + weight_index].to_uint())
                                        : (W_T)0;
                                }
                            }
                        }
                    }

                    conv_ddr_compute_oc:
                    for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                        const int oc = oc_base + oc_inner;
                        conv_ddr_compute_oh:
                        for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                            const int oh = oh_base + oh_inner;
                            conv_ddr_compute_ow:
                            for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
                                const int ow = ow_base + ow_inner;
                                conv_ddr_compute_ic:
                                for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IC_UNROLL
                                    if (oc < OUT_C && oh < OUT_H && ow < OUT_W) {
                                        conv_ddr_compute_kh:
                                        for (int kh = 0; kh < K; ++kh) {
                                            conv_ddr_compute_kw:
                                            for (int kw = 0; kw < K; ++kw) {
                                                const int tile_ih = oh_inner * STRIDE + kh;
                                                const int tile_iw = ow_inner * STRIDE + kw;
                                                acc_tile[oc_inner][oh_inner][ow_inner] +=
                                                    (ACC_T)input_tile[ic_inner][tile_ih][tile_iw] *
                                                    (ACC_T)conv_weight_tile[oc_inner][ic_inner][kh][kw];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                conv_ddr_store_output_oc:
                for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                    const int oc = oc_base + oc_inner;
                    conv_ddr_store_output_oh:
                    for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                        const int oh = oh_base + oh_inner;
                        conv_ddr_store_output_ow:
                        for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
                            const int ow = ow_base + ow_inner;
                            if (oc < OUT_C && oh < OUT_H && ow < OUT_W) {
                                output[(oc * OUT_H + oh) * OUT_W + ow] =
                                    (OUT_T)acc_tile[oc_inner][oh_inner][ow_inner];
                            }
                        }
                    }
                }
            }
        }
    }
}
'''


def _fpgai_conv_ddr_tile_template(template: str) -> str:
    parts = [part.strip() for part in template.split(",")]
    if len(parts) < 14:
        return template
    out_h = parts[3]
    out_w = parts[4]
    out_c = parts[5]
    in_c = parts[2]
    return ", ".join([*parts[:9], out_c, out_h, out_w, in_c, *parts[9:]])


def _fpgai_rewrite_conv_calls_for_ddr_tiled(source: str, graph) -> str:
    import re
    call_re = re.compile(
        r"conv2d(?P<tiled>_tiled)?\s*<(?P<template>[^>]*)>\s*\((?P<args>[^;]*)\)\s*;",
        re.MULTILINE | re.DOTALL,
    )
    offsets = _fpgai_ddr_weight_offsets_by_type(graph, "Conv")
    call_index = 0
    used = False

    def replace(match: re.Match[str]) -> str:
        nonlocal call_index, used
        args = _fpgai_split_cpp_args(match.group("args"))
        if len(args) < 2:
            return match.group(0)
        template = match.group("template") if match.group("tiled") else _fpgai_conv_ddr_tile_template(match.group("template"))
        weight_base, bias_base = offsets[call_index] if call_index < len(offsets) else (0, 0)
        call_index += 1
        used = True
        return (
            f"conv2d_ddr_tiled<{template}>("
            f"{args[0]}, {args[1]}, weights_mem, {weight_base}, {bias_base});"
        )

    rewritten = call_re.sub(replace, source)
    if used and "FPGAI real DDR-tiled Conv helper" not in rewritten:
        if "#include <ap_int.h>" not in rewritten:
            rewritten = rewritten.replace("#include <ap_axi_sdata.h>", "#include <ap_axi_sdata.h>\n#include <ap_int.h>")
        signature = 'extern "C" void '
        pos = rewritten.find(signature)
        if pos >= 0:
            rewritten = rewritten[:pos] + _fpgai_conv_ddr_tiled_helper_cpp() + "\n" + rewritten[pos:]
        else:
            rewritten = _fpgai_conv_ddr_tiled_helper_cpp() + "\n" + rewritten
    return rewritten

def _fpgai_insert_ddr_tiled_pragmas(source: str) -> str:
    marker = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
    replacement = (
        "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights\n"
        "#pragma HLS INTERFACE s_axilite port=weights_mem bundle=control\n"
        "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
    )
    if marker not in source:
        raise ValueError("Could not insert DDR-tiled weights_mem pragmas")
    return source.replace(marker, replacement, 1)

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

template<typename T, int N>
static void fpgai_store_ddr_vector(ap_uint<32>* weights_mem, int& offset, const T in[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        weights_mem[offset + i] = value_to_bits<T>(in[i]);
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
    elif mode == "ddr_tiled":
        runtime_signature = (
            f'extern "C" void {top_name}(\n'
            "    hls::stream<axis_t>& in_stream,\n"
            "    hls::stream<axis_t>& out_stream,\n"
            "    const ap_uint<32>* weights_mem\n"
            ") {"
        )
    else:
        runtime_signature = (
            f'extern "C" void {top_name}(\n'
            "    hls::stream<axis_t>& in_stream,\n"
            "    hls::stream<axis_t>& out_stream,\n"
            "    ap_uint<32>* weights_mem,\n"
            "    int mode\n"
            ") {"
        )

    if original_signature not in source:
        raise ValueError("Could not rewrite top signature for runtime weight mode")
    return source.replace(original_signature, runtime_signature, 1)


def _fpgai_runtime_load_block(graph, *, mode: str, resolved_semantics: str | None = None) -> str:
    specs = _fpgai_runtime_weight_specs(graph)
    resolved = str(resolved_semantics or "").strip().lower()
    export_enabled = resolved in {"bram_import_export_full", "uram_import_export_full"}
    storage_impl = "uram" if mode == "uram" or resolved.startswith("uram_import") else "bram"
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
        lines.extend([
            "    // FPGAI runtime weight command modes.",
            "    // mode 0: run inference with the already imported local weights.",
            "    // mode 1: import_weights copies the full runtime payload from weights_mem into local storage and returns.",
            "    static const int FPGAI_MODE_RUN_INFERENCE = 0;",
            "    static const int FPGAI_MODE_IMPORT_WEIGHTS = 1;",
        ])
        if export_enabled:
            lines.extend([
                "    // mode 2: export_weights copies current local weights back to weights_mem.",
                "    static const int FPGAI_MODE_EXPORT_WEIGHTS = 2;",
            ])
        if storage_impl == "uram":
            lines.extend([
                "    // Runtime-imported URAM weight storage.",
                "    // Weights are imported through weights_mem only when mode == FPGAI_MODE_IMPORT_WEIGHTS.",
            ])
        else:
            lines.extend([
                "    // Runtime-imported BRAM weight storage.",
                "    // Weights are imported through weights_mem only when mode == FPGAI_MODE_IMPORT_WEIGHTS.",
            ])
        for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
            lines.append(f"    // {op_type} {op_name}: local runtime W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}] in {storage_impl.upper()}")
            lines.append(f"    static {precision_tag}_wgt_t W{parameter_index}[{weight_count}];")
            lines.append(f"#pragma HLS BIND_STORAGE variable=W{parameter_index} type=ram_2p impl={storage_impl}")
            lines.append(f"    static {precision_tag}_bias_t B{parameter_index}[{bias_count}];")
            lines.append(f"#pragma HLS BIND_STORAGE variable=B{parameter_index} type=ram_2p impl={storage_impl}")
        lines.append("    if (mode == FPGAI_MODE_IMPORT_WEIGHTS) {")
        lines.append("        int fpgai_weight_offset = 0;")
        for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
            lines.append(f"        // import {op_type} {op_name}: W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}]")
            lines.append(
                f"        fpgai_load_ddr_vector<{precision_tag}_wgt_t, {weight_count}>(weights_mem, fpgai_weight_offset, W{parameter_index});"
            )
            lines.append(
                f"        fpgai_load_ddr_vector<{precision_tag}_bias_t, {bias_count}>(weights_mem, fpgai_weight_offset, B{parameter_index});"
            )
        lines.extend([
            "        return;",
            "    }",
            "",
        ])
        if export_enabled:
            lines.append("    if (mode == FPGAI_MODE_EXPORT_WEIGHTS) {")
            lines.append("        int fpgai_weight_offset = 0;")
            for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
                lines.append(f"        // export {op_type} {op_name}: W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}]")
                lines.append(
                    f"        fpgai_store_ddr_vector<{precision_tag}_wgt_t, {weight_count}>(weights_mem, fpgai_weight_offset, W{parameter_index});"
                )
                lines.append(
                    f"        fpgai_store_ddr_vector<{precision_tag}_bias_t, {bias_count}>(weights_mem, fpgai_weight_offset, B{parameter_index});"
                )
            lines.extend([
                "        return;",
                "    }",
                "",
            ])
        lines.extend([
            "    if (mode != FPGAI_MODE_RUN_INFERENCE) {",
            "        return;",
            "    }",
            "",
        ])
    return "\n".join(lines)


def _fpgai_insert_runtime_load_block(source: str, graph, *, mode: str, resolved_semantics: str | None = None) -> str:
    block = _fpgai_runtime_load_block(graph, mode=mode, resolved_semantics=resolved_semantics)
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
            "#pragma HLS INTERFACE s_axilite port=mode bundle=control\n"
            "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n"
            + block
            + "\n"
        )

    if pragma_marker not in source:
        raise ValueError("Could not insert runtime weight HLS pragmas")
    return source.replace(pragma_marker, replacement, 1)


def emit_top_cpp(*args, **kwargs):
    requested_mode = _fpgai_runtime_weight_mode(kwargs.get("weights_mode", "embedded"))
    if requested_mode not in {"stream", "streamed", "ddr", "dma_ddr", "uram", "ddr_tiled"}:
        return _fpgai_runtime_weight_previous_emit_top_cpp(*args, **kwargs)

    graph, top_name = _fpgai_runtime_recover_graph_top(args, kwargs)
    if graph is None or top_name is None:
        raise ValueError("Runtime weight top emission requires graph and top_name")

    if requested_mode == "ddr_tiled":
        updateed_kwargs = dict(kwargs)
        updateed_kwargs["weights_mode"] = "embedded"
        source = _fpgai_runtime_weight_previous_emit_top_cpp(*args, **updateed_kwargs)
        source = _fpgai_rewrite_runtime_signature(source, top_name=top_name, mode=requested_mode)
        source = _fpgai_insert_ddr_tiled_pragmas(source)
        source = _fpgai_rewrite_conv_calls_for_ddr_tiled(source, graph)
        source = _fpgai_rewrite_dense_calls_for_ddr_tiled(source, graph)
        planning_comment = (
            "// Requested weights mode: ddr_tiled\n"
            "// DDR-tiled Dense/Conv inference keeps full weights in weights_mem and materializes only tile-sized buffers on-chip.\n"
        )
        return planning_comment + source

    updateed_kwargs = dict(kwargs)
    updateed_kwargs["weights_mode"] = "embedded"
    source = _fpgai_runtime_weight_previous_emit_top_cpp(*args, **updateed_kwargs)
    source = _fpgai_insert_runtime_helpers(source)
    source = _fpgai_rewrite_runtime_signature(source, top_name=top_name, mode=requested_mode)
    notes = {}
    notes.update(_fpgai_notes_dict(kwargs.get("compile_plan")))
    notes.update(_fpgai_notes_dict(kwargs.get("memory_plan")))
    resolved_semantics = str(notes.get("resolved_weight_semantics", notes.get("memory_semantics_mode", ""))).strip().lower()
    source = _fpgai_insert_runtime_load_block(source, graph, mode=requested_mode, resolved_semantics=resolved_semantics)

    planning_comment = (
        f"// Requested weights mode: {requested_mode}\n"
        "// Non-embedded weight modes are represented in the memory/compile plan; "
        "generated HLS exposes explicit import/export command modes instead of reloading before every compute.\n"
    )
    return planning_comment + source



# FPGAI communication-edge annotation wrapper.
# The normal top emitter still implements raw AXI stream I/O.  This wrapper
# makes tensor-edge communication planning visible in generated HLS artifacts
# without claiming unsupported hardware compression decoders.
_fpgai_comm_edge_previous_emit_top_cpp = emit_top_cpp


def _fpgai_comm_plan_to_dict(plan):
    if plan is None:
        return {}
    if hasattr(plan, "to_dict"):
        data = plan.to_dict()
        return data if isinstance(data, dict) else {}
    return plan if isinstance(plan, dict) else {}


def _fpgai_comm_edge_macros(communication_plan) -> str:
    plan = _fpgai_comm_plan_to_dict(communication_plan)
    edges = plan.get("edges", []) if isinstance(plan, dict) else []
    if not edges:
        return ""

    lines = [
        "// FPGAI communication tensor-edge plan.",
        "// Compression codecs other than raw are modeled unless implemented_in_hls=true.",
    ]

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        tensor = str(edge.get("tensor_name", "tensor")).replace(" ", "_")
        kind = str(edge.get("notes", {}).get("kind", tensor)).replace(" ", "_").upper()
        direction = str(edge.get("direction", "?"))
        codec = str(edge.get("codec", edge.get("encoding", "raw")))
        precision_bits = edge.get("precision_bits")
        transfer_bytes = edge.get("transfer_bytes")
        size_bytes = edge.get("size_bytes")
        implemented = edge.get("implemented_in_hls")

        macro_prefix = "FPGAI_COMM_" + "".join(
            ch if ch.isalnum() else "_"
            for ch in kind
        )

        if precision_bits is not None:
            lines.append(f"#define {macro_prefix}_PRECISION_BITS {int(precision_bits)}")
        if transfer_bytes is not None:
            lines.append(f"#define {macro_prefix}_TRANSFER_BYTES {int(transfer_bytes)}")

        lines.append(
            f"//   tensor={tensor} kind={kind.lower()} direction={direction} "
            f"codec={codec} size_bytes={size_bytes} transfer_bytes={transfer_bytes} "
            f"implemented_in_hls={implemented}"
        )

    return "\n".join(lines) + "\n\n"


def emit_top_cpp(*args, **kwargs):
    communication_plan = kwargs.get("communication_plan")
    source = _fpgai_comm_edge_previous_emit_top_cpp(*args, **kwargs)
    macros = _fpgai_comm_edge_macros(communication_plan)
    if not macros:
        return source
    if "FPGAI communication tensor-edge plan" in source:
        return source
    return macros + source

# FPGAI static-weight storage wrapper.
# Compile-time/static BRAM and URAM modes must materialize an exact on-chip
# storage location, not only global const arrays that Vitis may map to LUTROM
# or registers.  Keep the generated parameter constants in fpgai_params.cpp,
# then copy them once into function-scope static arrays bound to BRAM/URAM.
_fpgai_static_weight_previous_emit_top_cpp = emit_top_cpp


def _fpgai_notes_dict(value):
    notes = getattr(value, "notes", None)
    if isinstance(notes, dict):
        return notes
    if isinstance(value, dict):
        maybe = value.get("notes", {})
        return maybe if isinstance(maybe, dict) else {}
    return {}


def _fpgai_static_weight_impl_from_kwargs(kwargs) -> str | None:
    memory_plan = kwargs.get("memory_plan")
    compile_plan = kwargs.get("compile_plan")
    notes = {}
    notes.update(_fpgai_notes_dict(compile_plan))
    notes.update(_fpgai_notes_dict(memory_plan))
    mode = str(notes.get("resolved_weight_semantics", notes.get("memory_semantics_mode", ""))).strip().lower()
    if mode == "bram_static":
        return "bram"
    if mode == "uram_static":
        return "uram"
    return None


def _fpgai_static_weight_init_block(graph, *, impl: str) -> str:
    specs = _fpgai_runtime_weight_specs(graph)
    if not specs:
        return ""

    lines = [
        f"    // FPGAI {impl}_static weight storage.",
        "    // Initial values are compile-time generated constants in fpgai_params.cpp.",
        f"    // The top function imports them once into local static {impl.upper()} arrays, then reuses them across runs.",
        "    static bool fpgai_static_weights_initialized = false;",
    ]

    for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
        lines.append(f"    // {op_type} {op_name}: local static W{parameter_index}[{weight_count}], B{parameter_index}[{bias_count}]")
        lines.append(f"    static {precision_tag}_wgt_t W{parameter_index}[{weight_count}];")
        lines.append(f"#pragma HLS BIND_STORAGE variable=W{parameter_index} type=ram_2p impl={impl}")
        lines.append(f"    static {precision_tag}_bias_t B{parameter_index}[{bias_count}];")
        lines.append(f"#pragma HLS BIND_STORAGE variable=B{parameter_index} type=ram_2p impl={impl}")

    lines.extend([
        "    if (!fpgai_static_weights_initialized) {",
    ])

    for parameter_index, precision_tag, weight_count, bias_count, op_type, op_name in specs:
        lines.append(f"        for (int i = 0; i < {weight_count}; ++i) {{")
        lines.append("#pragma HLS PIPELINE II=1")
        lines.append(f"            W{parameter_index}[i] = fpgai::W{parameter_index}[i];")
        lines.append("        }")
        lines.append(f"        for (int i = 0; i < {bias_count}; ++i) {{")
        lines.append("#pragma HLS PIPELINE II=1")
        lines.append(f"            B{parameter_index}[i] = fpgai::B{parameter_index}[i];")
        lines.append("        }")

    lines.extend([
        "        fpgai_static_weights_initialized = true;",
        "    }",
        "",
    ])
    return "\n".join(lines)


def _fpgai_insert_static_weight_block(source: str, graph, *, impl: str) -> str:
    if f"FPGAI {impl}_static weight storage" in source:
        return source
    marker = "#pragma HLS INTERFACE s_axilite port=return bundle=control\n\n"
    block = _fpgai_static_weight_init_block(graph, impl=impl)
    if not block:
        return source
    if marker not in source:
        raise ValueError("Could not insert static BRAM/URAM weight initialization block")
    return source.replace(marker, marker + block, 1)


def emit_top_cpp(*args, **kwargs):
    impl = _fpgai_static_weight_impl_from_kwargs(kwargs)
    source = _fpgai_static_weight_previous_emit_top_cpp(*args, **kwargs)
    if impl is None:
        return source

    requested_mode = _fpgai_runtime_weight_mode(kwargs.get("weights_mode", "embedded"))
    if requested_mode != "embedded":
        return source

    graph, _top_name = _fpgai_runtime_recover_graph_top(args, kwargs)
    if graph is None:
        raise ValueError("Static BRAM/URAM weight top emission requires graph")

    return _fpgai_insert_static_weight_block(source, graph, impl=impl)

# FPGAI input/output m_axi full movement wrapper.
# This keeps the existing AXI-stream/DMA path as the default, and rewrites the
# top interface only when the communication plan explicitly requests full
# m_axi import/export for network input/output arrays.
_fpgai_m_axi_io_previous_emit_top_cpp = emit_top_cpp


def _fpgai_edge_notes(communication_plan, kind: str) -> dict:
    kind = str(kind).lower()
    edges = getattr(communication_plan, "edges", []) or []
    for edge in edges:
        notes = getattr(edge, "notes", {}) or {}
        edge_kind = str(notes.get("kind", "")).lower()
        if edge_kind == kind or (kind == "input" and edge_kind in {"inputs", "activation_in"}) or (kind == "output" and edge_kind in {"outputs", "activation_out"}):
            return dict(notes)
    return {}


def _fpgai_cfg_get(raw, path: str, default=None):
    if not isinstance(raw, dict):
        return default
    current = raw
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _fpgai_io_movement_kind(kwargs, kind: str) -> str:
    communication_plan = kwargs.get("communication_plan")
    notes = _fpgai_edge_notes(communication_plan, kind)
    interface = str(notes.get("interface") or "").strip().lower().replace("-", "_")
    transport = str(notes.get("transport") or "").strip().lower().replace("-", "_")
    policy = str(notes.get("policy") or "").strip().lower().replace("-", "_")

    raw = kwargs.get("raw_cfg") or {}
    if kind == "input":
        prefix = "data_movement.inputs.import"
        direct_prefix = "data_movement.inputs"
        legacy_prefix = "data_movement.input.load"
    else:
        prefix = "data_movement.outputs.export"
        direct_prefix = "data_movement.outputs"
        legacy_prefix = "data_movement.output.store"

    interface = interface or str(
        _fpgai_cfg_get(
            raw,
            f"{prefix}.interface",
            _fpgai_cfg_get(raw, f"{direct_prefix}.interface", _fpgai_cfg_get(raw, f"{legacy_prefix}.interface", "")),
        )
        or ""
    ).strip().lower().replace("-", "_")
    transport = transport or str(
        _fpgai_cfg_get(
            raw,
            f"{prefix}.transport",
            _fpgai_cfg_get(raw, f"{direct_prefix}.transport", _fpgai_cfg_get(raw, f"{legacy_prefix}.transport", "")),
        )
        or ""
    ).strip().lower().replace("-", "_")
    policy = policy or str(
        _fpgai_cfg_get(
            raw,
            f"{prefix}.policy",
            _fpgai_cfg_get(raw, f"{direct_prefix}.policy", _fpgai_cfg_get(raw, f"{legacy_prefix}.policy", "")),
        )
        or ""
    ).strip().lower().replace("-", "_")

    if interface in {"m_axi", "maxi", "ddr"} and (policy in {"", "full"}):
        return "m_axi_full"
    if interface in {"m_axi", "maxi", "ddr"} and policy == "tiled":
        return "m_axi_tiled"
    if interface in {"axi_stream", "axis", "stream"} or transport in {"dma", "axi_dma"}:
        if policy == "tiled":
            return "axi_stream_tiled"
        return "axi_stream_full"
    return "axi_stream_full"


def _fpgai_io_tile_size(kwargs, kind: str) -> int | None:
    communication_plan = kwargs.get("communication_plan")
    notes = _fpgai_edge_notes(communication_plan, kind)
    value = notes.get("tile_size")
    raw = kwargs.get("raw_cfg") or {}
    if kind == "input":
        prefixes = ("data_movement.inputs.import", "data_movement.inputs", "data_movement.input.load")
    else:
        prefixes = ("data_movement.outputs.export", "data_movement.outputs", "data_movement.output.store")
    for prefix in prefixes:
        if value is None:
            value = _fpgai_cfg_get(raw, f"{prefix}.tile_size", None)
        tiled = _fpgai_cfg_get(raw, f"{prefix}.tiled", None)
        if value is None and isinstance(tiled, dict):
            value = tiled.get("tile_size", tiled.get("size", tiled.get("words")))
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _fpgai_ensure_ap_int(source: str) -> str:
    if "#include <ap_int.h>" in source:
        return source
    if "#include <ap_axi_sdata.h>" in source:
        return source.replace("#include <ap_axi_sdata.h>", "#include <ap_axi_sdata.h>\n#include <ap_int.h>", 1)
    return "#include <ap_int.h>\n" + source


def _fpgai_rewrite_signature_for_m_axi_io(source: str, *, input_m_axi: bool, output_m_axi: bool) -> str:
    if input_m_axi:
        source = source.replace("hls::stream<axis_t>& in_stream", "const ap_uint<32>* input_mem", 1)
        source = source.replace(
            "#pragma HLS INTERFACE axis port=in_stream\n",
            "#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input\n"
            "#pragma HLS INTERFACE s_axilite port=input_mem bundle=control\n",
            1,
        )
    if output_m_axi:
        source = source.replace("hls::stream<axis_t>& out_stream", "ap_uint<32>* output_mem", 1)
        source = source.replace(
            "#pragma HLS INTERFACE axis port=out_stream\n",
            "#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output\n"
            "#pragma HLS INTERFACE s_axilite port=output_mem bundle=control\n",
            1,
        )
    return source


def _fpgai_rewrite_input_stream_to_m_axi(source: str) -> str:
    import re

    decl = re.search(r"    (?P<typ>\w+) (?P<buf>layer_in)\[(?P<size>\d+)\];", source)
    if not decl:
        raise ValueError("Could not find input buffer declaration for m_axi input movement")
    typ = decl.group("typ")
    buf = decl.group("buf")
    size = decl.group("size")

    pattern = re.compile(
        r"    static const int FPGAI_ACT_BITS = (?P<bits>\d+);\n"
        r"    static const int FPGAI_ACT_PER_AXIS = (?P<per>\d+);\n"
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet = in_stream\.read\(\);\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                .*?\n"
        r"            \}\n"
        r"        \}\n"
        r"    \}\n",
        re.DOTALL,
    )
    replacement = (
        f"    static const int FPGAI_ACT_BITS = {32};\n"
        f"    static const int FPGAI_ACT_PER_AXIS = 1;\n"
        f"    // m_axi full input import: input_mem -> {buf}.\n"
        f"    for (int index = 0; index < {size}; ++index) {{\n"
        "#pragma HLS PIPELINE II=1\n"
        f"        {buf}[index] = bits_to_value<{typ}>(input_mem[index].to_uint());\n"
        "    }\n"
    )
    new_source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream input loop for m_axi input movement")
    return new_source


def _fpgai_rewrite_input_stream_to_m_axi_tiled(source: str, *, requested_tile_size: int | None = None) -> str:
    import re

    decl = re.search(r"    (?P<typ>\w+) (?P<buf>layer_in)\[(?P<size>\d+)\];", source)
    if not decl:
        raise ValueError("Could not find input buffer declaration for m_axi tiled input movement")
    typ = decl.group("typ")
    buf = decl.group("buf")
    size = int(decl.group("size"))
    tile = min(int(requested_tile_size or 64), max(1, size))

    pattern = re.compile(
        r"    static const int FPGAI_ACT_BITS = (?P<bits>\d+);\n"
        r"    static const int FPGAI_ACT_PER_AXIS = (?P<per>\d+);\n"
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet = in_stream\.read\(\);\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                .*?\n"
        r"            \}\n"
        r"        \}\n"
        r"    \}\n",
        re.DOTALL,
    )
    replacement = (
        "    static const int FPGAI_ACT_BITS = 32;\n"
        "    static const int FPGAI_ACT_PER_AXIS = 1;\n"
        f"    static const int FPGAI_INPUT_TILE_SIZE = {tile};\n"
        f"    {typ} input_tile[FPGAI_INPUT_TILE_SIZE];\n"
        "#pragma HLS BIND_STORAGE variable=input_tile type=ram_1p impl=bram\n"
        f"    // m_axi tiled input import: input_mem -> input_tile -> {buf}.\n"
        f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_INPUT_TILE_SIZE) {{\n"
        f"        int tile_count = ((tile_base + FPGAI_INPUT_TILE_SIZE) <= {size}) ? FPGAI_INPUT_TILE_SIZE : ({size} - tile_base);\n"
        "        for (int lane = 0; lane < FPGAI_INPUT_TILE_SIZE; ++lane) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "            if (lane < tile_count) {\n"
        f"                input_tile[lane] = bits_to_value<{typ}>(input_mem[tile_base + lane].to_uint());\n"
        "            }\n"
        "        }\n"
        "        for (int lane = 0; lane < FPGAI_INPUT_TILE_SIZE; ++lane) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "            if (lane < tile_count) {\n"
        f"                {buf}[tile_base + lane] = input_tile[lane];\n"
        "            }\n"
        "        }\n"
        "    }\n"
    )
    new_source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream input loop for m_axi tiled input movement")
    return new_source


def _fpgai_rewrite_output_stream_to_m_axi(source: str) -> str:
    import re

    pattern = re.compile(
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet;\n"
        r"        packet\.data = 0;\n"
        r"        packet\.keep = -1;\n"
        r"        packet\.strb = -1;\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                fpgai_pack_axis_value<(?P<typ>\w+), FPGAI_ACT_BITS>\(packet, (?P<buf>\w+)\[index\], lane\);\n"
        r"            \}\n"
        r"        \}\n"
        r"        packet\.last = \(base \+ FPGAI_ACT_PER_AXIS >= (?P=size)\) \? 1 : 0;\n"
        r"        out_stream\.write\(packet\);\n"
        r"    \}\n",
        re.DOTALL,
    )

    def replace(match):
        size = match.group("size")
        typ = match.group("typ")
        buf = match.group("buf")
        return (
            f"    // m_axi full output export: {buf} -> output_mem.\n"
            f"    for (int index = 0; index < {size}; ++index) {{\n"
            "#pragma HLS PIPELINE II=1\n"
            f"        output_mem[index] = value_to_bits<{typ}>({buf}[index]);\n"
            "    }\n"
        )

    new_source, count = pattern.subn(replace, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream output loop for m_axi output movement")
    return new_source


def _fpgai_rewrite_output_stream_to_m_axi_tiled(source: str, *, requested_tile_size: int | None = None) -> str:
    import re

    pattern = re.compile(
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet;\n"
        r"        packet\.data = 0;\n"
        r"        packet\.keep = -1;\n"
        r"        packet\.strb = -1;\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                fpgai_pack_axis_value<(?P<typ>\w+), FPGAI_ACT_BITS>\(packet, (?P<buf>\w+)\[index\], lane\);\n"
        r"            \}\n"
        r"        \}\n"
        r"        packet\.last = \(base \+ FPGAI_ACT_PER_AXIS >= (?P=size)\) \? 1 : 0;\n"
        r"        out_stream\.write\(packet\);\n"
        r"    \}\n",
        re.DOTALL,
    )

    def replace(match):
        size = int(match.group("size"))
        typ = match.group("typ")
        buf = match.group("buf")
        tile = min(int(requested_tile_size or 64), max(1, size))
        return (
            f"    static const int FPGAI_OUTPUT_TILE_SIZE = {tile};\n"
            f"    {typ} output_tile[FPGAI_OUTPUT_TILE_SIZE];\n"
            "#pragma HLS BIND_STORAGE variable=output_tile type=ram_1p impl=bram\n"
            f"    // m_axi tiled output export: {buf} -> output_tile -> output_mem.\n"
            f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_OUTPUT_TILE_SIZE) {{\n"
            f"        int tile_count = ((tile_base + FPGAI_OUTPUT_TILE_SIZE) <= {size}) ? FPGAI_OUTPUT_TILE_SIZE : ({size} - tile_base);\n"
            "        for (int lane = 0; lane < FPGAI_OUTPUT_TILE_SIZE; ++lane) {\n"
            "#pragma HLS PIPELINE II=1\n"
            "            if (lane < tile_count) {\n"
            f"                output_tile[lane] = {buf}[tile_base + lane];\n"
            "            }\n"
            "        }\n"
            "        for (int lane = 0; lane < FPGAI_OUTPUT_TILE_SIZE; ++lane) {\n"
            "#pragma HLS PIPELINE II=1\n"
            "            if (lane < tile_count) {\n"
            f"                output_mem[tile_base + lane] = value_to_bits<{typ}>(output_tile[lane]);\n"
            "            }\n"
            "        }\n"
            "    }\n"
        )

    new_source, count = pattern.subn(replace, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream output loop for m_axi tiled output movement")
    return new_source




def _fpgai_rewrite_input_stream_to_axis_tiled(source: str, *, requested_tile_size: int | None = None) -> str:
    import re

    decl = re.search(r"    (?P<typ>\w+) (?P<buf>layer_in)\[(?P<size>\d+)\];", source)
    if not decl:
        raise ValueError("Could not find input buffer declaration for AXI-stream tiled input movement")
    typ = decl.group("typ")
    buf = decl.group("buf")
    size = int(decl.group("size"))
    tile = min(int(requested_tile_size or 64), max(1, size))

    pattern = re.compile(
        r"    static const int FPGAI_ACT_BITS = (?P<bits>\d+);\n"
        r"    static const int FPGAI_ACT_PER_AXIS = (?P<per>\d+);\n"
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet = in_stream\.read\(\);\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                .*?\n"
        r"            \}\n"
        r"        \}\n"
        r"    \}\n",
        re.DOTALL,
    )
    replacement = (
        f"    static const int FPGAI_ACT_BITS = 32;\n"
        f"    static const int FPGAI_ACT_PER_AXIS = 1;\n"
        f"    static const int FPGAI_AXIS_INPUT_TILE_SIZE = {tile};\n"
        f"    {typ} input_tile[FPGAI_AXIS_INPUT_TILE_SIZE];\n"
        "#pragma HLS BIND_STORAGE variable=input_tile type=ram_1p impl=bram\n"
        f"    // AXI-stream tiled input import: in_stream -> input_tile -> {buf}.\n"
        f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_AXIS_INPUT_TILE_SIZE) {{\n"
        f"        int tile_count = ((tile_base + FPGAI_AXIS_INPUT_TILE_SIZE) <= {size}) ? FPGAI_AXIS_INPUT_TILE_SIZE : ({size} - tile_base);\n"
        "        for (int lane = 0; lane < FPGAI_AXIS_INPUT_TILE_SIZE; ++lane) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "            if (lane < tile_count) {\n"
        "                axis_t packet = in_stream.read();\n"
        f"                input_tile[lane] = fpgai_unpack_axis_value<{typ}, FPGAI_ACT_BITS>(packet, 0);\n"
        "            }\n"
        "        }\n"
        "        for (int lane = 0; lane < FPGAI_AXIS_INPUT_TILE_SIZE; ++lane) {\n"
        "#pragma HLS PIPELINE II=1\n"
        "            if (lane < tile_count) {\n"
        f"                {buf}[tile_base + lane] = input_tile[lane];\n"
        "            }\n"
        "        }\n"
        "    }\n"
    )
    new_source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream input loop for AXI-stream tiled input movement")
    return new_source


def _fpgai_rewrite_output_stream_to_axis_tiled(source: str, *, requested_tile_size: int | None = None) -> str:
    import re

    pattern = re.compile(
        r"    for \(int base = 0; base < (?P<size>\d+); base \+= FPGAI_ACT_PER_AXIS\) \{\n"
        r"#pragma HLS PIPELINE II=1\n"
        r"        axis_t packet;\n"
        r"        packet\.data = 0;\n"
        r"        packet\.keep = -1;\n"
        r"        packet\.strb = -1;\n"
        r"        for \(int lane = 0; lane < FPGAI_ACT_PER_AXIS; \+\+lane\) \{\n"
        r"#pragma HLS UNROLL\n"
        r"            int index = base \+ lane;\n"
        r"            if \(index < (?P=size)\) \{\n"
        r"                fpgai_pack_axis_value<(?P<typ>\w+), FPGAI_ACT_BITS>\(packet, (?P<buf>\w+)\[index\], lane\);\n"
        r"            \}\n"
        r"        \}\n"
        r"        packet\.last = \(base \+ FPGAI_ACT_PER_AXIS >= (?P=size)\) \? 1 : 0;\n"
        r"        out_stream\.write\(packet\);\n"
        r"    \}\n",
        re.DOTALL,
    )

    def replace(match):
        size = int(match.group("size"))
        typ = match.group("typ")
        buf = match.group("buf")
        tile = min(int(requested_tile_size or 64), max(1, size))
        return (
            f"    static const int FPGAI_AXIS_OUTPUT_TILE_SIZE = {tile};\n"
            f"    {typ} output_tile[FPGAI_AXIS_OUTPUT_TILE_SIZE];\n"
            "#pragma HLS BIND_STORAGE variable=output_tile type=ram_1p impl=bram\n"
            f"    // AXI-stream tiled output export: {buf} -> output_tile -> out_stream.\n"
            f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_AXIS_OUTPUT_TILE_SIZE) {{\n"
            f"        int tile_count = ((tile_base + FPGAI_AXIS_OUTPUT_TILE_SIZE) <= {size}) ? FPGAI_AXIS_OUTPUT_TILE_SIZE : ({size} - tile_base);\n"
            "        for (int lane = 0; lane < FPGAI_AXIS_OUTPUT_TILE_SIZE; ++lane) {\n"
            "#pragma HLS PIPELINE II=1\n"
            "            if (lane < tile_count) {\n"
            f"                output_tile[lane] = {buf}[tile_base + lane];\n"
            "            }\n"
            "        }\n"
            "        for (int lane = 0; lane < FPGAI_AXIS_OUTPUT_TILE_SIZE; ++lane) {\n"
            "#pragma HLS PIPELINE II=1\n"
            "            if (lane < tile_count) {\n"
            "                axis_t packet;\n"
            "                packet.data = 0;\n"
            "                packet.keep = -1;\n"
            "                packet.strb = -1;\n"
            f"                fpgai_pack_axis_value<{typ}, FPGAI_ACT_BITS>(packet, output_tile[lane], 0);\n"
            f"                packet.last = ((tile_base + lane + 1) >= {size}) ? 1 : 0;\n"
            "                out_stream.write(packet);\n"
            "            }\n"
            "        }\n"
            "    }\n"
        )

    new_source, count = pattern.subn(replace, source, count=1)
    if count != 1:
        raise ValueError("Could not rewrite AXI stream output loop for AXI-stream tiled output movement")
    return new_source


def _fpgai_apply_axis_tiled_io(source: str, *, input_kind: str, output_kind: str, input_tile_size: int | None = None, output_tile_size: int | None = None) -> str:
    if input_kind != "axi_stream_tiled" and output_kind != "axi_stream_tiled":
        return source
    if "FPGAI AXI-stream tiled input/output movement" in source:
        return source
    if input_kind == "axi_stream_tiled":
        source = _fpgai_rewrite_input_stream_to_axis_tiled(source, requested_tile_size=input_tile_size)
    if output_kind == "axi_stream_tiled":
        source = _fpgai_rewrite_output_stream_to_axis_tiled(source, requested_tile_size=output_tile_size)
    return "// FPGAI AXI-stream tiled input/output movement.\n" + source

def _fpgai_apply_m_axi_io(source: str, *, input_kind: str, output_kind: str, input_tile_size: int | None = None, output_tile_size: int | None = None) -> str:
    input_m_axi = input_kind in {"m_axi_full", "m_axi_tiled"}
    output_m_axi = output_kind in {"m_axi_full", "m_axi_tiled"}
    if not input_m_axi and not output_m_axi:
        return source
    if "FPGAI m_axi input/output movement" in source:
        return source
    source = _fpgai_ensure_ap_int(source)
    source = _fpgai_rewrite_signature_for_m_axi_io(source, input_m_axi=input_m_axi, output_m_axi=output_m_axi)
    if input_kind == "m_axi_full":
        source = _fpgai_rewrite_input_stream_to_m_axi(source)
    elif input_kind == "m_axi_tiled":
        source = _fpgai_rewrite_input_stream_to_m_axi_tiled(source, requested_tile_size=input_tile_size)
    if output_kind == "m_axi_full":
        source = _fpgai_rewrite_output_stream_to_m_axi(source)
    elif output_kind == "m_axi_tiled":
        source = _fpgai_rewrite_output_stream_to_m_axi_tiled(source, requested_tile_size=output_tile_size)
    return "// FPGAI m_axi input/output movement.\n" + source


def emit_top_cpp(*args, **kwargs):
    input_kind = _fpgai_io_movement_kind(kwargs, "input")
    output_kind = _fpgai_io_movement_kind(kwargs, "output")
    input_tile_size = _fpgai_io_tile_size(kwargs, "input")
    output_tile_size = _fpgai_io_tile_size(kwargs, "output")

    source = _fpgai_m_axi_io_previous_emit_top_cpp(*args, **kwargs)
    source = _fpgai_apply_m_axi_io(
        source,
        input_kind=input_kind,
        output_kind=output_kind,
        input_tile_size=input_tile_size,
        output_tile_size=output_tile_size,
    )
    return _fpgai_apply_axis_tiled_io(
        source,
        input_kind=input_kind,
        output_kind=output_kind,
        input_tile_size=input_tile_size,
        output_tile_size=output_tile_size,
    )


# FPGAI activation storage wrapper.
# Public activation storage is intentionally limited to BRAM/URAM.  This
# wrapper rewrites local activation buffer storage pragmas generated by the
# normal top emitter so that memory.storage.activations affects real HLS
# BIND_STORAGE artifacts.
_fpgai_activation_storage_previous_emit_top_cpp = emit_top_cpp


def _fpgai_activation_storage_impl(kwargs) -> str | None:
    raw = kwargs.get("raw_cfg") or {}
    memory_plan = kwargs.get("memory_plan")
    notes = _fpgai_notes_dict(memory_plan)
    value = str(notes.get("resolved_activation_storage", "") or "").strip().lower().replace("-", "_")
    if not value:
        value = str(_fpgai_cfg_get(raw, "memory.storage.activations", _fpgai_cfg_get(raw, "memory.activation_storage", "")) or "").strip().lower().replace("-", "_")
    aliases = {"block": "bram", "block_ram": "bram", "bram": "bram", "ultra": "uram", "ultra_ram": "uram", "uram": "uram"}
    return aliases.get(value)


def _fpgai_apply_activation_storage(source: str, impl: str | None) -> str:
    if impl not in {"bram", "uram"}:
        return source
    if f"FPGAI activation storage: {impl}" in source:
        return source
    import re
    pattern = re.compile(
        r"#pragma HLS BIND_STORAGE variable=(?P<name>layer_in|layer_\d+_out) type=ram_(?P<ports>\dp) impl=(?:bram|uram)"
    )
    source, count = pattern.subn(
        lambda m: f"#pragma HLS BIND_STORAGE variable={m.group('name')} type=ram_{m.group('ports')} impl={impl}",
        source,
    )
    if count == 0:
        # Keep this strict: if a user asked for activation storage, generated
        # source must contain real activation buffer binding sites.
        raise ValueError("Could not find activation buffer BIND_STORAGE pragmas to apply memory.storage.activations")
    return f"// FPGAI activation storage: {impl} local activation buffers.\n" + source


def emit_top_cpp(*args, **kwargs):
    source = _fpgai_activation_storage_previous_emit_top_cpp(*args, **kwargs)
    return _fpgai_apply_activation_storage(source, _fpgai_activation_storage_impl(kwargs))

# FPGAI readability wrapper.
# Keeps existing generation path intact and prepends an honest resolved-decision
# summary for researcher/contributor inspection when codegen.readability asks for it.
_fpgai_readability_previous_emit_top_cpp = emit_top_cpp


def _fpgai_readability_value(kwargs) -> str:
    raw = kwargs.get("raw_cfg") or {}
    value = str(_fpgai_cfg_get(raw, "codegen.readability", "high") or "high").strip().lower().replace("-", "_")
    return value if value in {"compact", "normal", "high", "debug"} else "high"


def _fpgai_plan_note(kwargs, key: str, default=""):
    for plan_key in ("memory_plan", "communication_plan", "compile_plan"):
        plan = kwargs.get(plan_key)
        notes = getattr(plan, "notes", None)
        if isinstance(notes, dict) and key in notes:
            return notes.get(key)
    return default


def _fpgai_readability_banner(kwargs, *, kind: str) -> str:
    level = _fpgai_readability_value(kwargs)
    if level == "compact":
        return ""
    raw = kwargs.get("raw_cfg") or {}
    weights_mode = str(kwargs.get("weights_mode", ""))
    memory_mode = str(_fpgai_plan_note(kwargs, "memory_semantics_mode", weights_mode))
    activation_storage = str(_fpgai_plan_note(kwargs, "resolved_activation_storage", _fpgai_cfg_get(raw, "memory.storage.activations", "bram")))
    pipeline_mode = str(_fpgai_cfg_get(raw, "pipeline.mode", "inference"))
    if level == "normal":
        return (
            "// FPGAI generated HLS top: "
            f"pipeline={pipeline_mode}, weights_mode={weights_mode}, memory_semantics={memory_mode}.\n"
        )
    input_kind = _fpgai_io_movement_kind(kwargs, "input") if kind == "inference" else "training_default"
    output_kind = _fpgai_io_movement_kind(kwargs, "output") if kind == "inference" else "training_default"
    return "\n".join([
        "// ============================================================",
        "// FPGAI generated HLS top",
        f"// Pipeline mode: {pipeline_mode}",
        f"// Codegen readability: {level}",
        f"// Weight mode: {weights_mode}",
        f"// Weight semantics: {memory_mode}",
        f"// Runtime import/export is command-driven; reload before each compute: false",
        f"// Activation storage: {activation_storage}",
        f"// Input movement: {input_kind}",
        f"// Output movement: {output_kind}",
        "// Sections: includes/types, runtime constants, storage, import/export helpers, compute helpers, top dispatch.",
        "// ============================================================",
        "",
    ])


def emit_top_cpp(*args, **kwargs):
    source = _fpgai_readability_previous_emit_top_cpp(*args, **kwargs)
    banner = _fpgai_readability_banner(kwargs, kind="inference")
    if banner and "FPGAI generated HLS top" not in source[:512]:
        return banner + source
    return source
