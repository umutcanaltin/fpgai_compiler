from __future__ import annotations

from typing import List
from fpgai.ir.graph import Graph


def emit_weights_runtime_h(g: Graph) -> str:
    lines: List[str] = []
    lines.append("#pragma once")
    lines.append('#include <hls_stream.h>')
    lines.append('#include <ap_axi_sdata.h>')
    lines.append('#include "fpgai_types.h"')
    lines.append('#include "fpgai_params.h"')
    lines.append("")
    lines.append("typedef ap_axis<32,0,0,0> axis32_t;")
    lines.append("")
    lines.append("namespace fpgai {")
    lines.append("")
    lines.append("// Runtime weight injection (stream / ddr-preload mode)")
    lines.append("//")
    lines.append("// Expected payload order is graph order:")
    lines.append("//   - For Conv layers: W[OUT_C][IN_C][K][K], then B[OUT_C]")
    lines.append("//   - For Dense layers: W[OUT][IN], then B[OUT]")
    lines.append("//")
    lines.append("// Values are sent as 32-bit float words and cast to FPGAI numeric types.")
    lines.append("")
    lines.append("void fpgai_load_weights_from_stream(hls::stream<axis32_t>& inStream);")
    lines.append("")
    lines.append("} // namespace fpgai")
    lines.append("")
    return "\n".join(lines)