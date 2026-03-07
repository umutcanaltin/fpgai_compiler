from fpgai.ir.graph import Graph

def emit_params_h_stub(graph: Graph) -> str:
    lines = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append("namespace fpgai {")
    lines.append("")

    weight_idx = 0
    
    for op in graph.ops:
        if op.op_type in ["Conv", "Dense"]:
            w_name = None
            b_name = None

            # 1. Resolve Weight/Bias Names based on Op Type
            if op.op_type == "Dense":
                # Dense params are in attributes (canonicalized)
                w_name = op.attrs.get("weight")
                b_name = op.attrs.get("bias")
            elif op.op_type == "Conv":
                # Conv params are in inputs (standard ONNX)
                if len(op.inputs) > 1: w_name = op.inputs[1]
                if len(op.inputs) > 2: b_name = op.inputs[2]

            # 2. Emit Weight Declaration
            if w_name:
                w_tensor = graph.get_tensor(w_name)
                # Fallback: check constants if tensor spec is missing
                if not w_tensor and w_name in graph.constants:
                     # Create a dummy spec or just use the array shape
                     shape = graph.constants[w_name].shape
                     total = 1
                     for d in shape: total *= d
                     lines.append(f"    extern const wgt_t W{weight_idx}[{total}];")
                elif w_tensor:
                    total = 1
                    for d in w_tensor.shape: total *= d
                    lines.append(f"    extern const wgt_t W{weight_idx}[{total}];")
            
            # 3. Emit Bias Declaration
            if b_name:
                b_tensor = graph.get_tensor(b_name)
                if not b_tensor and b_name in graph.constants:
                    shape = graph.constants[b_name].shape
                    total = 1
                    for d in shape: total *= d
                    lines.append(f"    extern const bias_t B{weight_idx}[{total}];")
                elif b_tensor:
                    total = 1
                    for d in b_tensor.shape: total *= d
                    lines.append(f"    extern const bias_t B{weight_idx}[{total}];")
            else:
                # Always emit a dummy bias if missing, to satisfy template arguments
                lines.append(f"    extern const bias_t B{weight_idx}[1];")
            
            weight_idx += 1

    lines.append("}")
    return "\n".join(lines)