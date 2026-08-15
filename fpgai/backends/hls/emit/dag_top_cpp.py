from __future__ import annotations

"""Branch-aware inference top emission using FPGAI tensor liveness."""

from typing import Any, Mapping

from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation
from fpgai.backends.hls.emit.types_h import _default_precision, _op_precision_from_attrs, _spec_to_ap


def _flat_size(shape: Any) -> int:
    dims = tuple(int(x) for x in (shape or (1,)))
    if len(dims) > 1 and dims[0] == 1:
        dims = dims[1:]
    total = 1
    for dim in dims:
        if dim <= 0:
            raise ValueError("HLSDAG001: dynamic/non-positive tensor dimension")
        total *= dim
    return total


def _tensor_words(graph: Any, name: str) -> int:
    spec = graph.get_tensor(name) if hasattr(graph, "get_tensor") else None
    if spec is None or not getattr(spec, "shape", None):
        raise ValueError(f"HLSDAG002: missing static shape for tensor {name!r}")
    return _flat_size(spec.shape)


def _resolved_tensor_types(graph: Any, raw_cfg: Mapping[str, Any] | None) -> tuple[dict[str, str], str]:
    raw = dict(raw_cfg or {})
    defaults = _default_precision(raw)
    activation_default = _spec_to_ap(defaults["activation"])
    types: dict[str, str] = {str(name): activation_default for name in getattr(graph, "inputs", []) or []}
    for op in getattr(graph, "ops", []) or []:
        precision = _op_precision_from_attrs(op, defaults)
        output_type = _spec_to_ap(precision["activation"])
        for name in getattr(op, "outputs", []) or []:
            types[str(name)] = output_type
    return types, _spec_to_ap(defaults["accum"])


def emit_dag_top_cpp(
    graph: Any,
    *,
    top_name: str,
    weights_mode: str,
    raw_cfg: Mapping[str, Any] | None = None,
    external_composition_plan: Any = None,
    tensor_liveness: Mapping[str, Any] | None = None,
    buffer_allocation: Mapping[str, Any] | None = None,
) -> str:
    if str(weights_mode).strip().lower() != "embedded":
        raise ValueError("HLSDAG003: DAG inference currently requires embedded weights mode")
    if len(getattr(graph, "inputs", []) or []) != 1 or len(getattr(graph, "outputs", []) or []) != 1:
        raise ValueError("HLSDAG004: maintained DAG top currently requires one graph input and one graph output")

    allocation = dict(buffer_allocation or build_hls_buffer_allocation(
        graph, raw_cfg=raw_cfg, tensor_liveness=tensor_liveness
    ))
    tensor_to_buffer = dict(allocation.get("tensor_to_buffer", {}))
    tensor_types, accumulator_type = _resolved_tensor_types(graph, raw_cfg)
    defaults = _default_precision(dict(raw_cfg or {}))
    act_spec = defaults["activation"]
    act_bits = int(act_spec.get("total_bits", 16)) if isinstance(act_spec, Mapping) else 16
    if act_bits <= 0 or act_bits > 32 or 32 % act_bits != 0:
        raise ValueError(f"HLSDAG005: activation width {act_bits} is not supported by 32-bit AXIS packing")
    act_per_axis = 32 // act_bits

    lines: list[str] = [
        "#include <hls_stream.h>",
        "#include <ap_axi_sdata.h>",
        '#include "fpgai_types.h"',
        '#include "fpgai_params.h"',
        '#include "layers/activations.h"',
        '#include "layers/dense.h"',
        '#include "layers/conv.h"',
        "",
        "typedef ap_axis<32, 0, 0, 0> axis_t;",
        "using namespace fpgai;",
        "",
    ]
    if external_composition_plan is not None:
        from fpgai.implementations.hls_composition import package_declarations
        lines.extend(package_declarations(external_composition_plan))
        lines.append("")

    lines.extend([
        "template<typename T, int VALUE_BITS>",
        "static inline T fpgai_unpack_axis_value(axis_t packet, int lane) {",
        "#pragma HLS INLINE",
        "    ap_uint<VALUE_BITS> raw = packet.data.range(((lane + 1) * VALUE_BITS) - 1, lane * VALUE_BITS);",
        "    T value;",
        "    value.range(VALUE_BITS - 1, 0) = raw;",
        "    return value;",
        "}",
        "",
        "template<typename T, int VALUE_BITS>",
        "static inline void fpgai_pack_axis_value(axis_t& packet, T value, int lane) {",
        "#pragma HLS INLINE",
        "    ap_uint<VALUE_BITS> raw = value.range(VALUE_BITS - 1, 0);",
        "    packet.data.range(((lane + 1) * VALUE_BITS) - 1, lane * VALUE_BITS) = raw;",
        "}",
        "",
        f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {{',
        "#pragma HLS INTERFACE axis port=in_stream",
        "#pragma HLS INTERFACE axis port=out_stream",
        "#pragma HLS INTERFACE s_axilite port=return bundle=control",
        "",
        f"    static const int FPGAI_ACT_BITS = {act_bits};",
        f"    static const int FPGAI_ACT_PER_AXIS = {act_per_axis};",
        "",
    ])

    for slot in allocation.get("slots", []) or []:
        lines.append(
            f"    {slot['cpp_type']} {slot['name']}[{int(slot['words'])}];"
        )
        lines.append(
            f"    // FPGAI_BUFFER_SLOT {slot['name']} tensors={','.join(str(x) for x in slot.get('tensors', []))}"
        )
    lines.append("")

    input_name = str(graph.inputs[0])
    input_buffer = tensor_to_buffer[input_name]
    input_type = tensor_types[input_name]
    input_words = _tensor_words(graph, input_name)
    lines.extend([
        f"    // FPGAI_BUFFER_PROVENANCE buffer={input_buffer} tensor={input_name}",
        f"    for (int base = 0; base < {input_words}; base += FPGAI_ACT_PER_AXIS) {{",
        "#pragma HLS PIPELINE II=1",
        "        axis_t packet = in_stream.read();",
        "        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {",
        "#pragma HLS UNROLL",
        "            int index = base + lane;",
        f"            if (index < {input_words}) {input_buffer}[index] = fpgai_unpack_axis_value<{input_type}, FPGAI_ACT_BITS>(packet, lane);",
        "        }",
        "    }",
        "",
    ])

    parameter_index = 0
    for index, op in enumerate(graph.ops):
        runtime_inputs = [str(x) for x in (getattr(op, "inputs", []) or []) if str(x) not in getattr(graph, "constants", {})]
        outputs = [str(x) for x in (getattr(op, "outputs", []) or [])]
        binding = external_composition_plan.binding_for_node(op.name) if external_composition_plan is not None else None
        if binding is not None:
            from fpgai.implementations.hls_integration import parse_hls_abi, HLSFlatArrayABI
            abi = parse_hls_abi(binding.contract)
            lines.append(f"    // DAG node {index}: {op.op_type} ({op.name})")
            for name in outputs:
                lines.append(f"    // FPGAI_BUFFER_PROVENANCE buffer={tensor_to_buffer[name]} tensor={name}")
            if isinstance(abi, HLSFlatArrayABI):
                if len(runtime_inputs) != 1 or len(outputs) != 1:
                    raise RuntimeError(f"HLSDAG007: external flat_array_v1 node {op.name!r} requires one runtime tensor input/output")
                output_name = outputs[0]; output_buffer = tensor_to_buffer[output_name]; output_type = tensor_types[output_name]; output_words = _tensor_words(graph, output_name)
                input_name_for_op = runtime_inputs[0]
                from fpgai.implementations.hls_composition import emit_external_call
                lines.extend(emit_external_call(binding, current_buffer=tensor_to_buffer[input_name_for_op], current_type=tensor_types[input_name_for_op], output_buffer=output_buffer, output_type=output_type))
            else:
                from fpgai.implementations.hls_composition import emit_external_tensor_ports_call
                lines.extend(emit_external_tensor_ports_call(
                    binding,
                    input_buffers={name: tensor_to_buffer[name] for name in runtime_inputs},
                    input_types={name: tensor_types[name] for name in runtime_inputs},
                    output_buffers={name: tensor_to_buffer[name] for name in outputs},
                    output_types={name: tensor_types[name] for name in outputs},
                ))
            lines.append("")
            continue

        if len(outputs) != 1:
            raise RuntimeError(f"HLSDAG006: node {op.name!r} must have exactly one output in the current built-in DAG profile")
        output_name = outputs[0]
        output_buffer = tensor_to_buffer[output_name]
        output_type = tensor_types[output_name]
        output_words = _tensor_words(graph, output_name)
        lines.append(f"    // DAG node {index}: {op.op_type} ({op.name})")
        lines.append(f"    // FPGAI_BUFFER_PROVENANCE buffer={output_buffer} tensor={output_name}")

        if op.op_type in {"Relu", "Sigmoid", "LeakyRelu", "Identity", "Flatten", "Reshape"}:
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG008: unary node {op.name!r} requires one runtime tensor input")
            source_name = runtime_inputs[0]
            source_buffer = tensor_to_buffer[source_name]
            source_type = tensor_types[source_name]
            if op.op_type == "Relu":
                lines.append(f"    relu_typed<{output_words}, {source_type}, {output_type}>({source_buffer}, {output_buffer});")
            elif op.op_type == "Sigmoid":
                lines.append(f"    sigmoid_typed<{output_words}, {source_type}, {output_type}, {accumulator_type}>({source_buffer}, {output_buffer});")
            elif op.op_type == "LeakyRelu":
                alpha = float((getattr(op, "attrs", {}) or {}).get("alpha", 0.1))
                lines.append(f"    leaky_relu_typed<{output_words}, {source_type}, {output_type}, {accumulator_type}>({source_buffer}, {output_buffer}, ({accumulator_type}){alpha:.17g});")
            else:
                lines.append(f"    reshape_copy_typed<{output_words}, {source_type}, {output_type}>({source_buffer}, {output_buffer});")
        elif op.op_type == "Dense":
            if len(runtime_inputs) != 1:
                raise RuntimeError(f"HLSDAG012: Dense node {op.name!r} requires one runtime activation input")
            source_name = runtime_inputs[0]
            source_words = _tensor_words(graph, source_name)
            lines.append(
                f"    dense_out_in<{source_words}, {output_words}, {tensor_types[source_name]}, {output_type}, op{index}_wgt_t, op{index}_bias_t, op{index}_acc_t>("
                f"{tensor_to_buffer[source_name]}, {output_buffer}, W{parameter_index}, B{parameter_index});"
            )
            parameter_index += 1
        elif op.op_type == "Conv":
            if len(runtime_inputs) != 1 or len(getattr(op, "inputs", []) or []) < 2:
                raise RuntimeError(f"HLSDAG013: Conv node {op.name!r} requires one runtime input plus embedded weights")
            source_name = runtime_inputs[0]
            in_spec = graph.get_tensor(source_name); out_spec = graph.get_tensor(output_name)
            in_shape = tuple(int(x) for x in in_spec.shape); out_shape = tuple(int(x) for x in out_spec.shape)
            if len(in_shape) == 4 and in_shape[0] == 1: in_shape = in_shape[1:]
            if len(out_shape) == 4 and out_shape[0] == 1: out_shape = out_shape[1:]
            if len(in_shape) != 3 or len(out_shape) != 3:
                raise RuntimeError(f"HLSDAG014: Conv node {op.name!r} requires NCHW/CHW static shapes")
            ic, ih, iw = in_shape; oc, oh, ow = out_shape
            weight_name = op.inputs[1]
            weight = (getattr(graph, "constants", {}) or {}).get(weight_name)
            weight_shape = tuple(int(x) for x in getattr(weight, "shape", ()))
            if len(weight_shape) != 4 or weight_shape[2] != weight_shape[3]:
                raise RuntimeError(f"HLSDAG015: Conv node {op.name!r} requires square embedded kernels")
            kernel = weight_shape[2]; strides = (getattr(op, "attrs", {}) or {}).get("strides", [1,1]); pads=(getattr(op, "attrs", {}) or {}).get("pads", [0,0,0,0])
            if int(strides[0]) != int(strides[1]) or not (int(pads[0]) == int(pads[1]) == int(pads[2]) == int(pads[3])):
                raise RuntimeError(f"HLSDAG016: Conv node {op.name!r} requires symmetric stride/padding")
            lines.append(
                f"    conv2d<{ih}, {iw}, {ic}, {oh}, {ow}, {oc}, {kernel}, {int(strides[0])}, {int(pads[0])}, {tensor_types[source_name]}, {output_type}, op{index}_wgt_t, op{index}_bias_t, op{index}_acc_t>("
                f"{tensor_to_buffer[source_name]}, {output_buffer}, "
                f"reinterpret_cast<const op{index}_wgt_t*>(W{parameter_index}), B{parameter_index});"
            )
            parameter_index += 1
        elif op.op_type == "Add":
            if len(runtime_inputs) != 2:
                raise RuntimeError(f"HLSDAG009: Add node {op.name!r} requires exactly two runtime tensor inputs")
            left_name, right_name = runtime_inputs
            left_words = _tensor_words(graph, left_name)
            right_words = _tensor_words(graph, right_name)
            if left_words != right_words or left_words != output_words:
                raise RuntimeError(f"HLSDAG010: Add node {op.name!r} requires equal flattened input/output sizes")
            lines.append(
                f"    add_vec_typed<{output_words}, {tensor_types[left_name]}, {tensor_types[right_name]}, {output_type}, {accumulator_type}>("
                f"{tensor_to_buffer[left_name]}, {tensor_to_buffer[right_name]}, {output_buffer});"
            )
        else:
            raise RuntimeError(
                f"HLSDAG011: operator {op.op_type!r} is not yet supported by the branch-aware HLS emitter"
            )
        lines.append("")

    output_name = str(graph.outputs[0])
    output_buffer = tensor_to_buffer[output_name]
    output_type = tensor_types[output_name]
    output_words = _tensor_words(graph, output_name)
    lines.extend([
        f"    // FPGAI_BUFFER_PROVENANCE buffer={output_buffer} tensor={output_name}",
        f"    for (int base = 0; base < {output_words}; base += FPGAI_ACT_PER_AXIS) {{",
        "#pragma HLS PIPELINE II=1",
        "        axis_t packet;",
        "        packet.data = 0; packet.keep = -1; packet.strb = -1; packet.last = 0;",
        "        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {",
        "#pragma HLS UNROLL",
        "            int index = base + lane;",
        f"            if (index < {output_words}) fpgai_pack_axis_value<{output_type}, FPGAI_ACT_BITS>(packet, {output_buffer}[index], lane);",
        "        }",
        f"        packet.last = (base + FPGAI_ACT_PER_AXIS >= {output_words}) ? 1 : 0;",
        "        out_stream.write(packet);",
        "    }",
        "}",
        "",
    ])
    return "\n".join(lines)
