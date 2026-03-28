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
            d = lp.to_dict() if hasattr(lp, "to_dict") else (lp if isinstance(lp, dict) else {})
            if d.get("node_name"):
                out[d["node_name"]] = d
        return out
    if isinstance(compile_plan, dict):
        return {lp["node_name"]: lp for lp in compile_plan.get("layer_plans", []) if isinstance(lp, dict) and lp.get("node_name")}
    return {}


def _comm_map(communication_plan: Any) -> Dict[str, Dict[str, Any]]:
    if communication_plan is None:
        return {}
    if hasattr(communication_plan, "edges"):
        out = {}
        for e in communication_plan.edges:
            d = e.to_dict() if hasattr(e, "to_dict") else (e if isinstance(e, dict) else {})
            if d.get("tensor_name"):
                out[d["tensor_name"]] = d
        return out
    if isinstance(communication_plan, dict):
        return {e["tensor_name"]: e for e in communication_plan.get("edges", []) if isinstance(e, dict) and e.get("tensor_name")}
    return {}


def _memory_map(memory_plan: Any) -> Dict[str, Dict[str, Any]]:
    if memory_plan is None:
        return {}
    if hasattr(memory_plan, "placements"):
        out = {}
        for p in memory_plan.placements:
            d = p.to_dict() if hasattr(p, "to_dict") else (p if isinstance(p, dict) else {})
            if d.get("tensor_name"):
                out[d["tensor_name"]] = d
        return out
    if isinstance(memory_plan, dict):
        return {p["tensor_name"]: p for p in memory_plan.get("placements", []) if isinstance(p, dict) and p.get("tensor_name")}
    return {}


def _strip_batch(shape) -> Tuple[int, ...]:
    if shape is None:
        return tuple()
    shape = tuple(int(x) for x in shape)
    if len(shape) > 1 and shape[0] == 1:
        return tuple(shape[1:])
    return shape


def _flat_size(shape) -> int:
    return int(np.prod(shape)) if shape else 1


def _as_chw(shape3: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if len(shape3) != 3:
        raise ValueError(f"Expected 3D shape, got {shape3}")
    c, h, w = shape3
    return int(c), int(h), int(w)


def _conv_out_dim(in_dim: int, k: int, stride: int, pad_total: int, dilation: int = 1) -> int:
    return ((in_dim + pad_total - dilation * (k - 1) - 1) // stride) + 1


def _infer_out_shape(graph: Graph, op, curr_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    out_name = op.outputs[0]
    out_spec = graph.get_tensor(out_name)
    if out_spec is not None and getattr(out_spec, "shape", None):
        return _strip_batch(tuple(int(v) for v in out_spec.shape))

    if op.op_type == "Conv":
        c_in, h_in, w_in = _as_chw(curr_shape)
        out_c = None
        k_h = 3
        k_w = 3

        if len(op.inputs) > 1:
            w_name = op.inputs[1]
            w_t = graph.get_tensor(w_name)
            if w_t is not None and getattr(w_t, "shape", None):
                ws = tuple(int(v) for v in w_t.shape)
                if len(ws) == 4:
                    out_c, k_h, k_w = ws[0], ws[2], ws[3]
            elif hasattr(graph, "constants") and w_name in graph.constants:
                ws = tuple(int(v) for v in graph.constants[w_name].shape)
                if len(ws) == 4:
                    out_c, k_h, k_w = ws[0], ws[2], ws[3]

        strides = op.attrs.get("strides", [1, 1])
        pads = op.attrs.get("pads", [0, 0, 0, 0])
        dilations = op.attrs.get("dilations", [1, 1])

        sh, sw = int(strides[0]), int(strides[1])
        pt, pl, pb, pr = [int(x) for x in pads]
        dh, dw = int(dilations[0]), int(dilations[1])

        h_out = _conv_out_dim(h_in, k_h, sh, pt + pb, dh)
        w_out = _conv_out_dim(w_in, k_w, sw, pl + pr, dw)
        return (int(out_c), int(h_out), int(w_out))

    if op.op_type in ("MaxPool", "AvgPool"):
        c_in, h_in, w_in = _as_chw(curr_shape)
        kernel = op.attrs.get("kernel_shape", [2, 2])
        strides = op.attrs.get("strides", [2, 2])
        pads = op.attrs.get("pads", [0, 0, 0, 0])

        kh, kw = int(kernel[0]), int(kernel[1])
        sh, sw = int(strides[0]), int(strides[1])
        pt, pl, pb, pr = [int(x) for x in pads]

        h_out = _conv_out_dim(h_in, kh, sh, pt + pb, 1)
        w_out = _conv_out_dim(w_in, kw, sw, pl + pr, 1)
        return (int(c_in), int(h_out), int(w_out))

    if op.op_type in ("Relu", "Sigmoid", "LeakyRelu", "Add", "BatchNormalization", "Softmax"):
        return tuple(curr_shape)

    if op.op_type in ("Flatten", "Reshape"):
        return (_flat_size(curr_shape),)

    if op.op_type == "Dense":
        out_f = op.attrs.get("out_features", op.attrs.get("out"))
        if out_f is None and "weight" in op.attrs:
            w_name = op.attrs["weight"]
            w_t = graph.get_tensor(w_name)
            if w_t is not None and getattr(w_t, "shape", None):
                out_f = tuple(int(v) for v in w_t.shape)[0]
        return (int(out_f),)

    raise ValueError(f"Could not infer output shape for op {op.name} ({op.op_type})")


def _placement_comment(mem_info: Dict[str, Any] | None) -> str:
    if not mem_info:
        return "unknown"
    return f"{mem_info.get('region', 'unknown')} / size={mem_info.get('size_bytes', 'unknown')} bytes"


def _comm_comment(comm_info: Dict[str, Any] | None) -> str:
    if not comm_info:
        return "unknown"
    return f"{comm_info.get('direction', 'unknown')} / {comm_info.get('encoding', 'raw')}"


def _emit_plan_comments(lines, indent: str, lp: Dict[str, Any], out_mem: Dict[str, Any] | None, out_comm: Dict[str, Any] | None):
    lines.append(f"{indent}//   precision_mode : {lp.get('precision_mode')}")
    lines.append(f"{indent}//   act_bits       : {lp.get('act_bits')}")
    lines.append(f"{indent}//   weight_bits    : {lp.get('weight_bits')}")
    lines.append(f"{indent}//   tile           : {lp.get('tile', {})}")
    lines.append(f"{indent}//   unroll         : {lp.get('unroll', {})}")
    lines.append(f"{indent}//   pipeline_ii    : {lp.get('pipeline_ii')}")
    lines.append(f"{indent}//   weight_mode    : {lp.get('weight_mode')}")
    lines.append(f"{indent}//   activation_mode: {lp.get('activation_mode')}")
    lines.append(f"{indent}//   buffering      : {lp.get('buffering')}")
    lines.append(f"{indent}//   output placement: {_placement_comment(out_mem)}")
    lines.append(f"{indent}//   output communication: {_comm_comment(out_comm)}")


def _emit_storage_pragmas(lines, var_name: str, mem_info: Dict[str, Any] | None):
    region = str((mem_info or {}).get("region", "BRAM")).upper()
    if region == "URAM":
        lines.append(f"#pragma HLS BIND_STORAGE variable={var_name} type=ram_1p impl=uram")
    else:
        lines.append(f"#pragma HLS BIND_STORAGE variable={var_name} type=ram_1p impl=bram")


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

    input_shape = (1, 1024)
    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x and x.shape:
            input_shape = tuple(int(v) for v in x.shape)

    lines = []
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
    lines.append('#include "layers/dense.h"')
    lines.append('#include "layers/conv.h"')
    lines.append('#include "layers/pool.h"')
    lines.append('#include "layers/activations.h"')
    lines.append('#if defined(FPGAI_DEBUG_DUMP) && !defined(__SYNTHESIS__)')
    lines.append('#include <fstream>')
    lines.append('#endif')
    lines.append("")
    lines.append("typedef ap_axis<32, 0, 0, 0> axis_t;")
    lines.append("using namespace fpgai;")
    lines.append("")
    lines.append("inline act_t bits_to_act(unsigned int bits) { union { unsigned int i; float f; } c; c.i = bits; return (act_t)c.f; }")
    lines.append("inline unsigned int act_to_bits(act_t val) { union { unsigned int i; float f; } c; c.f = (float)val; return c.i; }")
    lines.append("")
    lines.append("#if defined(FPGAI_DEBUG_DUMP) && !defined(__SYNTHESIS__)")
    lines.append("static inline void fpgai_dump_tensor(const char* path, const act_t* data, int n) {")
    lines.append("    std::ofstream f(path, std::ios::binary);")
    lines.append("    for (int i = 0; i < n; i++) {")
    lines.append("        float v = (float)data[i];")
    lines.append("        f.write(reinterpret_cast<const char*>(&v), sizeof(float));")
    lines.append("    }")
    lines.append("}")
    lines.append("#endif")
    lines.append("")

    lines.append(f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {{')
    lines.append("#pragma HLS INTERFACE axis port=in_stream")
    lines.append("#pragma HLS INTERFACE axis port=out_stream")
    lines.append("#pragma HLS INTERFACE s_axilite port=return bundle=control")
    lines.append("")

    input_tensor_name = graph.inputs[0] if graph.inputs else "input"
    input_comm = comm_by_tensor.get(input_tensor_name)
    input_mem = mem_by_tensor.get(input_tensor_name)

    lines.append(f"    // input transfer: {_comm_comment(input_comm)}")
    lines.append(f"    // input placement: {_placement_comment(input_mem)}")

    input_shape_nobatch = _strip_batch(input_shape)
    input_flat = _flat_size(input_shape_nobatch)
    curr_buf = "layer_in"
    curr_shape = input_shape_nobatch

    lines.append(f"    act_t {curr_buf}[{input_flat}];")
    _emit_storage_pragmas(lines, curr_buf, input_mem)
    lines.append(f"    for(int i=0; i<{input_flat}; i++) {{")
    lines.append("        axis_t temp = in_stream.read();")
    lines.append(f"        {curr_buf}[i] = bits_to_act(temp.data.to_uint());")
    lines.append("    }")
    lines.append("")

    param_idx = 0
    layer_idx = 0

    for op in graph.ops:
        lp = plan_by_name.get(op.name, {})
        out_name = op.outputs[0]
        out_shape = _infer_out_shape(graph, op, curr_shape)
        out_flat = _flat_size(out_shape)
        out_buf = f"layer_{layer_idx}_out"

        out_mem = mem_by_tensor.get(out_name)
        out_comm = comm_by_tensor.get(out_name)

        lines.append(f"    // Layer {layer_idx}: {op.op_type} ({op.name})")
        _emit_plan_comments(lines, "    ", lp, out_mem, out_comm)
        lines.append(f"    act_t {out_buf}[{out_flat}];")
        _emit_storage_pragmas(lines, out_buf, out_mem)

        if op.op_type == "Conv":
            c_in, h_in, w_in = _as_chw(curr_shape)
            c_out, h_out, w_out = _as_chw(out_shape)

            strides = op.attrs.get("strides", [1, 1])
            pads = op.attrs.get("pads", [0, 0, 0, 0])

            if len(op.inputs) > 1:
                w_name = op.inputs[1]
                w_t = graph.get_tensor(w_name)
                if w_t is not None and getattr(w_t, "shape", None):
                    ws = tuple(int(v) for v in w_t.shape)
                elif hasattr(graph, "constants") and w_name in graph.constants:
                    ws = tuple(int(v) for v in graph.constants[w_name].shape)
                else:
                    raise ValueError(f"Conv weights not found for op {op.name}")
                k_h = int(ws[2])
            else:
                raise ValueError(f"Conv op {op.name} missing weight input")

            stride_h = int(strides[0])
            pad = int(pads[0]) if pads else 0

            tile = lp.get("tile", {})
            if tile:
                lines.append(f"    // conv planned tiling: oh={tile.get('oh', 'NA')} ow={tile.get('ow', 'NA')} oc={tile.get('oc', 'NA')}")

            lines.append(f"    conv2d<{h_in}, {w_in}, {c_in}, {h_out}, {w_out}, {c_out}, {k_h}, {stride_h}, {pad}>({curr_buf}, {out_buf}, W{param_idx}, B{param_idx});")
            param_idx += 1

        elif op.op_type == "Dense":
            in_size = _flat_size(curr_shape)
            out_size = _flat_size(out_shape)
            tile = lp.get("tile", {})
            if tile:
                lines.append(f"    // dense planned tiling: in={tile.get('in', 'NA')} out={tile.get('out', 'NA')}")
            lines.append(f"    dense_out_in<{in_size}, {out_size}>({curr_buf}, {out_buf}, W{param_idx}, B{param_idx});")
            param_idx += 1

        elif op.op_type == "Relu":
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")
            lines.append(f"    relu_inplace<{out_flat}>({out_buf});")

        elif op.op_type == "LeakyRelu":
            alpha = float(op.attrs.get("alpha", 0.1))
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")
            lines.append(f"    leaky_relu_inplace<{out_flat}>({out_buf}, (act_t){alpha});")

        elif op.op_type == "Sigmoid":
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")
            lines.append(f"    sigmoid_inplace<{out_flat}>({out_buf});")

        elif op.op_type == "Softmax":
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")
            lines.append(f"    softmax_inplace<{out_flat}>({out_buf});")

        elif op.op_type in ("MaxPool", "AvgPool"):
            c_in, h_in, w_in = _as_chw(curr_shape)
            c_out, h_out, w_out = _as_chw(out_shape)
            kernel = op.attrs.get("kernel_shape", [2, 2])
            strides = op.attrs.get("strides", [2, 2])

            k_h = int(kernel[0])
            stride_h = int(strides[0])

            fn = "maxpool2d" if op.op_type == "MaxPool" else "avgpool2d"
            lines.append(f"    {fn}<{h_in}, {w_in}, {c_in}, {k_h}, {stride_h}, {h_out}, {w_out}>({curr_buf}, {out_buf});")

        elif op.op_type == "Flatten":
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")

        elif op.op_type == "Reshape":
            if len(curr_shape) == 3 and len(out_shape) == 1:
                c_in, h_in, w_in = _as_chw(curr_shape)
                lines.append(f"    for(int c=0; c<{c_in}; c++) {{")
                lines.append(f"        for(int h=0; h<{h_in}; h++) {{")
                lines.append(f"            for(int w=0; w<{w_in}; w++) {{")
                lines.append(f"                int src_idx = (h * {w_in} + w) * {c_in} + c;")
                lines.append(f"                int dst_idx = (c * {h_in} + h) * {w_in} + w;")
                lines.append(f"                {out_buf}[dst_idx] = {curr_buf}[src_idx];")
                lines.append("            }")
                lines.append("        }")
                lines.append("    }")
            else:
                lines.append(f"    reshape_copy<{out_flat}>({curr_buf}, {out_buf});")
        else:
            lines.append(f"    for(int j=0; j<{out_flat}; j++) {out_buf}[j] = {curr_buf}[j];")

        lines.append("#if defined(FPGAI_DEBUG_DUMP) && !defined(__SYNTHESIS__)")
        lines.append(f'    fpgai_dump_tensor("{op.name}.bin", {out_buf}, {out_flat});')
        lines.append("#endif")
        lines.append("")

        curr_buf = out_buf
        curr_shape = out_shape
        layer_idx += 1

    output_tensor_name = graph.outputs[0] if graph.outputs else "output"
    output_comm = comm_by_tensor.get(output_tensor_name)
    output_mem = mem_by_tensor.get(output_tensor_name)

    out_flat = _flat_size(curr_shape)
    lines.append(f"    // output transfer: {_comm_comment(output_comm)}")
    lines.append(f"    // output placement: {_placement_comment(output_mem)}")
    lines.append(f"    for(int i=0; i<{out_flat}; i++) {{")
    lines.append("        axis_t temp;")
    lines.append(f"        temp.data = act_to_bits({curr_buf}[i]);")
    lines.append("        temp.keep = -1; temp.strb = -1;")
    lines.append(f"        temp.last = (i == {out_flat}-1);")
    lines.append("        out_stream.write(temp);")
    lines.append("    }")
    lines.append("}")

    return "\n".join(lines)