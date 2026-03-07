from fpgai.ir.graph import Graph
import numpy as np

def emit_params_cpp(graph: Graph) -> str:
    lines = []
    lines.append('#include "fpgai_params.h"')
    lines.append("namespace fpgai {")
    lines.append("")

    weight_idx = 0

    for op in graph.ops:
        if op.op_type in ["Conv", "Dense"]:
            w_name = None
            b_name = None

            # 1. Resolve Names
            if op.op_type == "Dense":
                w_name = op.attrs.get("weight")
                b_name = op.attrs.get("bias")
            elif op.op_type == "Conv":
                if len(op.inputs) > 1: w_name = op.inputs[1]
                if len(op.inputs) > 2: b_name = op.inputs[2]

            # 2. Emit Weight Data
            if w_name and w_name in graph.constants:
                w_data = graph.constants[w_name]
                data = w_data.flatten()
                
                lines.append(f"    const wgt_t W{weight_idx}[{data.size}] = {{")
                
                # Optimized string generation
                s_vals = [f"{x:.6f}" for x in data]
                for i in range(0, len(s_vals), 10):
                    chunk = ", ".join(s_vals[i:i+10])
                    suffix = "," if (i + 10 < len(s_vals)) else ""
                    lines.append(f"        {chunk}{suffix}")
                lines.append("    };")
            else:
                lines.append(f"    const wgt_t W{weight_idx}[1] = {{ 0 }};")

            # 3. Emit Bias Data
            if b_name and b_name in graph.constants:
                b_data = graph.constants[b_name]
                data = b_data.flatten()
                
                lines.append(f"    const bias_t B{weight_idx}[{data.size}] = {{")
                
                s_vals = [f"{x:.6f}" for x in data]
                for i in range(0, len(s_vals), 10):
                    chunk = ", ".join(s_vals[i:i+10])
                    suffix = "," if (i + 10 < len(s_vals)) else ""
                    lines.append(f"        {chunk}{suffix}")
                lines.append("    };")
            else:
                lines.append(f"    const bias_t B{weight_idx}[1] = {{ 0 }};")

            weight_idx += 1
            lines.append("")

    lines.append("}")
    return "\n".join(lines)