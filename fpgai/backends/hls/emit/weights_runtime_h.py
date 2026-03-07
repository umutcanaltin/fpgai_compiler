from __future__ import annotations

from typing import List
from fpgai.ir.graph import Graph


def emit_weights_runtime_h(g: Graph) -> str:
    dense_ops = [op for op in g.ops if op.op_type == "Dense"]

    lines: List[str] = []
    lines.append("#pragma once")
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
    lines.append("")
    lines.append("// Runtime weight injection (stream mode)")
    lines.append("void fpgai_load_weights_from_stream(hls::stream<axis32_t>& inStream);")
    lines.append("")
    lines.append("// Weight storage is declared in fpgai_params.h as extern arrays.")
    lines.append("// Definitions live in src/weights_runtime.cpp for stream mode.")
    lines.append("")
    return "\n".join(lines)
