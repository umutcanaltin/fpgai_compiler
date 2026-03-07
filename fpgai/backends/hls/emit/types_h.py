from fpgai.ir.graph import Graph

def emit_types_h(graph: Graph, *, top_name: str) -> str:
    return """#pragma once
#include <ap_fixed.h>

namespace fpgai {

// Precision configuration
typedef ap_fixed<16, 6> act_t;
typedef ap_fixed<16, 6> wgt_t;
typedef ap_fixed<24, 10> bias_t;

// Accumulator: Wider type to prevent overflow during matrix mul
typedef ap_fixed<24, 10> acc_t; 
typedef ap_fixed<24, 10> accum_t; // alias for compatibility

} // namespace fpgai
"""