from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from fpgai.ir.graph import Graph


def _plan_map(compile_plan: Any) -> Dict[str, Dict[str, Any]]:
    if compile_plan is None:
        return {}

    if hasattr(compile_plan, "layer_plans"):
        out = {}
        for lp in compile_plan.layer_plans:
            if hasattr(lp, "to_dict"):
                d = lp.to_dict()
            elif isinstance(lp, dict):
                d = lp
            else:
                d = {"node_name": getattr(lp, "node_name", None)}
            name = d.get("node_name")
            if name:
                out[name] = d
        return out

    if isinstance(compile_plan, dict):
        out = {}
        for lp in compile_plan.get("layer_plans", []):
            if isinstance(lp, dict) and lp.get("node_name"):
                out[lp["node_name"]] = lp
        return out

    return {}


def _comm_map(communication_plan: Any) -> Dict[str, Dict[str, Any]]:
    if communication_plan is None:
        return {}

    if hasattr(communication_plan, "edges"):
        out = {}
        for e in communication_plan.edges:
            if hasattr(e, "to_dict"):
                d = e.to_dict()
            elif isinstance(e, dict):
                d = e
            else:
                d = {"tensor_name": getattr(e, "tensor_name", None)}
            name = d.get("tensor_name")
            if name:
                out[name] = d
        return out

    if isinstance(communication_plan, dict):
        out = {}
        for e in communication_plan.get("edges", []):
            if isinstance(e, dict) and e.get("tensor_name"):
                out[e["tensor_name"]] = e
        return out

    return {}


def _memory_map(memory_plan: Any) -> Dict[str, Dict[str, Any]]:
    if memory_plan is None:
        return {}

    if hasattr(memory_plan, "placements"):
        out = {}
        for p in memory_plan.placements:
            if hasattr(p, "to_dict"):
                d = p.to_dict()
            elif isinstance(p, dict):
                d = p
            else:
                d = {"tensor_name": getattr(p, "tensor_name", None)}
            name = d.get("tensor_name")
            if name:
                out[name] = d
        return out

    if isinstance(memory_plan, dict):
        out = {}
        for p in memory_plan.get("placements", []):
            if isinstance(p, dict) and p.get("tensor_name"):
                out[p["tensor_name"]] = p
        return out

    return {}


def _strip_batch(shape) -> Tuple[int, ...]:
    if shape is None:
        return tuple()
    shape = tuple(int(x) for x in shape)
    if len(shape) > 1 and shape[0] == 1:
        return tuple(shape[1:])
    return shape


def _flat_size(shape) -> int:
    if not shape:
        return 1
    return int(np.prod(shape))


def _as_chw(shape3: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if len(shape3) != 3:
        raise ValueError(f"Expected 3D shape, got {shape3}")
    c, h, w = shape3
    return int(c), int(h), int(w)


def _as_hwc_for_kernel(shape3: Tuple[int, int, int]) -> Tuple[int, int, int]:
    c, h, w = _as_chw(shape3)
    return h, w, c


def emit_top_cpp(
    graph: Graph,
    *,
    top_name: str,
    weights_mode: str,
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
) -> str:
    plan_by_name = _plan_map(compile_plan)
    comm_by_tensor = _comm_map(communication_plan)
    mem_by_tensor = _memory_map(memory_plan)

    in_total_size = 1024
    input_shape = (1, in_total_size)
    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x and x.shape:
            input_shape = tuple(int(v) for v in x.shape)
            in_total_size = int(np.prod(x.shape))

    runtime_weights = weights_mode in ("stream", "ddr")

    lines = []
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
    if runtime_weights:
        lines.append('#include "weights_runtime.h"')
    lines.append('#include "layers/dense.h"')
    lines.append('#include "layers/conv.h"')
    lines.append('#include "layers/pool.h"')
    lines.append('#include "layers/activations.h"')
    lines.append("")
    lines.append("typedef ap_axis<32, 0, 0, 0> axis_t;")
    lines.append("using namespace fpgai;")
    lines.append("")
    lines.append("inline act_t bits_to_act(unsigned int bits) { union { unsigned int i; float f; } c; c.i = bits; return (act_t)c.f; }")
    lines.append("inline unsigned int act_to_bits(act_t val) { union { unsigned int i; float f; } c; c.f = (float)val; return c.i; }")
    lines.append("")

    lines.append("// ------------------------------------------------------------")
    lines.append("// FPGAI generated top")
    lines.append(f"// top_name    : {top_name}")
    lines.append(f"// weights_mode: {weights_mode}")

    if graph.inputs:
        in_name = graph.inputs[0]
        if in_name in comm_by_tensor:
            c = comm_by_tensor[in_name]
            lines.append(f"// input_comm  : {c.get('direction')} / {c.get('encoding')} / axi={c.get('axi_word_bits')}")
        if in_name in mem_by_tensor:
            m = mem_by_tensor[in_name]
            lines.append(f"// input_mem   : {m.get('region')} ({m.get('size_bytes')} bytes)")

    if graph.outputs:
        out_name = graph.outputs[0]
        if out_name in comm_by_tensor:
            c = comm_by_tensor[out_name]
            lines.append(f"// output_comm : {c.get('direction')} / {c.get('encoding')} / axi={c.get('axi_word_bits')}")
        if out_name in mem_by_tensor:
            m = mem_by_tensor[out_name]
            lines.append(f"// output_mem  : {m.get('region')} ({m.get('size_bytes')} bytes)")

    if runtime_weights:
        lines.append("// runtime weights: enabled")
        lines.append("// mode == 0 : preload weights from weight_stream")
        lines.append("// mode == 1 : run inference")
    else:
        lines.append("// runtime weights: disabled (embedded parameters)")

    lines.append("// ------------------------------------------------------------")
    lines.append("")

    if runtime_weights:
        lines.append(
            f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, '
            f'hls::stream<axis_t>& out_stream, '
            f'hls::stream<axis_t>& weight_stream, '
            f'int mode) {{'
        )
        lines.append(f'#pragma HLS INTERFACE axis port=in_stream')
        lines.append(f'#pragma HLS INTERFACE axis port=out_stream')
        lines.append(f'#pragma HLS INTERFACE axis port=weight_stream')
        lines.append(f'#pragma HLS INTERFACE s_axilite port=mode bundle=control')
        lines.append(f'#pragma HLS INTERFACE s_axilite port=return bundle=control')
        lines.append("")
        lines.append("    // mode 0: preload runtime weights")
        lines.append("    if (mode == 0) {")
        lines.append("        fpgai_load_weights_from_stream(weight_stream);")
        lines.append("        return;")
        lines.append("    }")
        lines.append("")
        lines.append("    // mode 1: inference")
        lines.append("")
    else:
        lines.append(f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {{')
        lines.append(f'#pragma HLS INTERFACE axis port=in_stream')
        lines.append(f'#pragma HLS INTERFACE axis port=out_stream')
        lines.append(f'#pragma HLS INTERFACE s_axilite port=return bundle=control')
        lines.append("")

    if graph.inputs:
        in_name = graph.inputs[0]
        if in_name in comm_by_tensor:
            c = comm_by_tensor[in_name]
            lines.append(f"    // input transfer: {c.get('direction')} / encoding={c.get('encoding')} / unpack_in_pl={c.get('unpack_in_pl')}")
        if in_name in mem_by_tensor:
            m = mem_by_tensor[in_name]
            lines.append(f"    // input placement: {m.get('region')} / size={m.get('size_bytes')} bytes")

    lines.append(f"    act_t layer_in[{in_total_size}];")
    lines.append(f"    for(int i=0; i<{in_total_size}; i++) {{")
    lines.append("        axis_t temp = in_stream.read();")
    lines.append("        layer_in[i] = bits_to_act(temp.data.to_uint());")
    lines.append("    }")
    lines.append("")

    curr_buf = "layer_in"
    curr_shape = _strip_batch(input_shape)
    weight_idx = 0

    for i, op in enumerate(graph.ops):
        lp = plan_by_name.get(op.name, {})
        lines.append(f"    // Layer {i}: {op.op_type} ({op.name})")

        if lp:
            lines.append(f"    //   precision_mode : {lp.get('precision_mode')}")
            lines.append(f"    //   act_bits       : {lp.get('act_bits')}")
            lines.append(f"    //   weight_bits    : {lp.get('weight_bits')}")
            lines.append(f"    //   tile           : {lp.get('tile', {})}")
            lines.append(f"    //   unroll         : {lp.get('unroll', {})}")
            lines.append(f"    //   pipeline_ii    : {lp.get('pipeline_ii')}")
            lines.append(f"    //   weight_mode    : {lp.get('weight_mode')}")
            lines.append(f"    //   activation_mode: {lp.get('activation_mode')}")
            lines.append(f"    //   buffering      : {lp.get('buffering')}")

        out_name = op.outputs[0]
        out_spec = graph.get_tensor(out_name)
        out_shape = _strip_batch(tuple(int(v) for v in out_spec.shape))
        out_flat_size = _flat_size(out_shape)

        if op.op_type in ["Flatten", "Reshape"]:
            curr_flat_size = _flat_size(curr_shape)
            if out_flat_size <= 1:
                out_flat_size = curr_flat_size

        if out_name in mem_by_tensor:
            m = mem_by_tensor[out_name]
            lines.append(f"    //   output placement: {m.get('region')} / size={m.get('size_bytes')} bytes")
        if out_name in comm_by_tensor:
            c = comm_by_tensor[out_name]
            lines.append(f"    //   output communication: {c.get('direction')} / {c.get('encoding')}")

        next_buf = f"layer_{i}_out"
        lines.append(f"    act_t {next_buf}[{out_flat_size}];")

        if op.op_type == "Dense":
            in_flat = _flat_size(curr_shape)
            out_flat = _flat_size(out_shape)
            cast_str = f"(const wgt_t (*)[{in_flat}])"

            lines.append(f"    // dense weights source index: W{weight_idx}, B{weight_idx}")
            if lp:
                lines.append(f"    // dense planned tiling: in={lp.get('tile', {}).get('in')} out={lp.get('tile', {}).get('out')}")

            lines.append(
                f"    dense_out_in<{in_flat}, {out_flat}>({curr_buf}, {next_buf}, {cast_str}W{weight_idx}, B{weight_idx});"
            )
            weight_idx += 1

        elif op.op_type == "Conv":
            k = int(op.attrs.get("kernel_shape", [3, 3])[0])
            pads = op.attrs.get("pads", [0, 0, 0, 0])
            stride = int(op.attrs.get("strides", [1, 1])[0])

            if len(curr_shape) != 3 or len(out_shape) != 3:
                raise ValueError(
                    f"Conv op {op.name} expects 3D feature maps after batch strip, got curr_shape={curr_shape}, out_shape={out_shape}"
                )

            H, W, C = _as_hwc_for_kernel(curr_shape)
            Ho, Wo, Co = _as_hwc_for_kernel(out_shape)

            lines.append(f"    // conv weights source index: W{weight_idx}, B{weight_idx}")
            lines.append(f"    // conv tensor order: source CHW={curr_shape} -> kernel HWC=({H}, {W}, {C})")
            if lp:
                lines.append(
                    f"    // conv planned tiling: oh={lp.get('tile', {}).get('oh')} "
                    f"ow={lp.get('tile', {}).get('ow')} oc={lp.get('tile', {}).get('oc')}"
                )

            lines.append(f"    conv2d<{H}, {W}, {C}, {Ho}, {Wo}, {Co}, {k}, {stride}, {int(pads[0])}>")
            lines.append(f"          ({curr_buf}, {next_buf}, W{weight_idx}, B{weight_idx});")
            weight_idx += 1

        elif op.op_type in ["MaxPool", "AvgPool"]:
            k = int(op.attrs.get("kernel_shape", [2, 2])[0])
            stride = int(op.attrs.get("strides", [2, 2])[0])

            if len(curr_shape) != 3 or len(out_shape) != 3:
                raise ValueError(
                    f"{op.op_type} op {op.name} expects 3D feature maps after batch strip, got curr_shape={curr_shape}, out_shape={out_shape}"
                )

            H, W, C = _as_hwc_for_kernel(curr_shape)
            Ho, Wo, Co = _as_hwc_for_kernel(out_shape)

            lines.append(f"    // pool tensor order: source CHW={curr_shape} -> kernel HWC=({H}, {W}, {C})")

            if op.op_type == "MaxPool":
                lines.append(f"    maxpool2d<{H}, {W}, {C}, {k}, {stride}, {Ho}, {Wo}>({curr_buf}, {next_buf});")
            else:
                lines.append(f"    avgpool2d<{H}, {W}, {C}, {k}, {stride}, {Ho}, {Wo}>({curr_buf}, {next_buf});")

        elif op.op_type == "Relu":
            lines.append(f"    for(int j=0; j<{out_flat_size}; j++) {next_buf}[j] = {curr_buf}[j];")
            lines.append(f"    relu_inplace<{out_flat_size}>({next_buf});")

        elif op.op_type == "Softmax":
            lines.append(f"    for(int j=0; j<{out_flat_size}; j++) {next_buf}[j] = {curr_buf}[j];")
            lines.append(f"    softmax_inplace<{out_flat_size}>({next_buf});")

        elif op.op_type in ["Flatten", "Reshape"]:
            lines.append(f"    for(int j=0; j<{out_flat_size}; j++) {next_buf}[j] = {curr_buf}[j];")

        else:
            lines.append(f"    // WARNING: Unknown Op {op.op_type}, skipping...")

        curr_buf = next_buf
        curr_shape = out_shape
        lines.append("")

    out_final_size = _flat_size(curr_shape)

    if graph.outputs:
        out_name = graph.outputs[0]
        if out_name in comm_by_tensor:
            c = comm_by_tensor[out_name]
            lines.append(f"    // output transfer: {c.get('direction')} / encoding={c.get('encoding')}")
        if out_name in mem_by_tensor:
            m = mem_by_tensor[out_name]
            lines.append(f"    // output placement: {m.get('region')} / size={m.get('size_bytes')} bytes")

    lines.append(f"    // Write Outputs")
    lines.append(f"    for(int i=0; i<{out_final_size}; i++) {{")
    lines.append("        axis_t temp;")
    lines.append(f"        temp.data = act_to_bits({curr_buf}[i]);")
    lines.append(f"        temp.keep = -1; temp.strb = -1;")
    lines.append(f"        temp.last = (i == {out_final_size}-1);")
    lines.append("        out_stream.write(temp);")
    lines.append("    }")
    lines.append("}")
    return "\n".join(lines)