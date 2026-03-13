from __future__ import annotations

from typing import Any
from fpgai.ir.graph import Graph


def emit_model_inst_cpp(graph: Graph, compile_plan: Any = None) -> str:
    return r'''#include "dense.h"
#include "conv.h"
#include "pool.h"
#include "activations.h"

namespace fpgai {

// Intentionally empty.
//
// Layer kernels are header-defined templates, so explicit
// instantiations are not required here.
//
// This avoids signature mismatches when kernel template
// parameter lists change (for example, when moving from
// global numeric types to per-layer typed kernels).

} // namespace fpgai
'''