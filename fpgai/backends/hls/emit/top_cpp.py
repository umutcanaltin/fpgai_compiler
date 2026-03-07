from __future__ import annotations
from fpgai.ir.graph import Graph
import numpy as np

def emit_top_cpp(graph: Graph, *, top_name: str, weights_mode: str) -> str:
    # 1. Determine Global Input Size
    in_total_size = 1024
    if graph.inputs:
        x = graph.get_tensor(graph.inputs[0])
        if x: in_total_size = int(np.prod(x.shape))
    
    lines = []
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
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

    lines.append(f'extern "C" void {top_name}(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {{')
    lines.append(f'#pragma HLS INTERFACE axis port=in_stream')
    lines.append(f'#pragma HLS INTERFACE axis port=out_stream')
    lines.append(f'#pragma HLS INTERFACE s_axilite port=return bundle=control')
    lines.append("")

    # --- Input Buffer ---
    lines.append(f"    act_t layer_in[{in_total_size}];")
    lines.append(f"    for(int i=0; i<{in_total_size}; i++) {{")
    lines.append("        axis_t temp = in_stream.read();")
    lines.append("        layer_in[i] = bits_to_act(temp.data.to_uint());")
    lines.append("    }")
    lines.append("")

    curr_buf = "layer_in"
    curr_shape = graph.get_tensor(graph.inputs[0]).shape if graph.inputs else (1, in_total_size)
    if len(curr_shape) > 1 and curr_shape[0] == 1: curr_shape = curr_shape[1:]

    weight_idx = 0

    # --- Iterate Layers ---
    for i, op in enumerate(graph.ops):
        lines.append(f"    // Layer {i}: {op.op_type}")
        
        # Determine Output Size
        out_name = op.outputs[0]
        out_spec = graph.get_tensor(out_name)
        out_shape = out_spec.shape[1:] if out_spec.shape[0] == 1 else out_spec.shape
        out_flat_size = int(np.prod(out_shape))
        
        # FIX: Flatten/Reshape Preservation
        # If shape inference failed (size 0) or it's a Reshape, assume simple pass-through
        if op.op_type in ["Flatten", "Reshape"]:
             curr_flat_size = int(np.prod(curr_shape))
             if out_flat_size <= 1: 
                 out_flat_size = curr_flat_size
        
        next_buf = f"layer_{i}_out"
        lines.append(f"    act_t {next_buf}[{out_flat_size}];")

        if op.op_type == "Dense":
            in_flat = int(np.prod(curr_shape))
            out_flat = int(np.prod(out_shape))
            cast_str = f"(const wgt_t (*)[{in_flat}])"
            lines.append(f"    dense_out_in<{in_flat}, {out_flat}>({curr_buf}, {next_buf}, {cast_str}W{weight_idx}, B{weight_idx});")
            weight_idx += 1

        elif op.op_type == "Conv":
            k = op.attrs.get('kernel_shape', [3, 3])[0]
            pads = op.attrs.get('pads', [0, 0, 0, 0])
            stride = op.attrs.get('strides', [1, 1])[0]
            H, W, C = curr_shape
            Ho, Wo, Co = out_shape
            
            # FIX: No cast needed, we updated conv.h to take (wgt_t*)
            lines.append(f"    conv2d<{H}, {W}, {C}, {Ho}, {Wo}, {Co}, {k}, {stride}, {pads[0]}>")
            lines.append(f"          ({curr_buf}, {next_buf}, W{weight_idx}, B{weight_idx});")
            weight_idx += 1

        elif op.op_type in ["MaxPool", "AvgPool"]:
            k = op.attrs.get('kernel_shape', [2, 2])[0]
            stride = op.attrs.get('strides', [2, 2])[0]
            H, W, C = curr_shape
            Ho, Wo, Co = out_shape
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

    # --- Write Output ---
    out_final_size = int(np.prod(curr_shape))
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