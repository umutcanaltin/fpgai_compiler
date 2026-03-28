from __future__ import annotations

from typing import Any, Dict


def _cp_get(compile_plan: Any) -> Dict[str, Any]:
    if compile_plan is None:
        return {}
    if hasattr(compile_plan, "to_dict"):
        return compile_plan.to_dict()
    if isinstance(compile_plan, dict):
        return compile_plan
    return {}


def emit_types_h(graph, *, top_name: str, compile_plan: Any = None) -> str:
    cp = _cp_get(compile_plan)
    notes = cp.get("notes", {}) or {}
    layer_plans = cp.get("layer_plans", []) or []

    act_bits = int(notes.get("act_bits", 16) or 16)
    act_int_bits = int(notes.get("act_int_bits", 6) or 6)
    weight_bits = int(notes.get("weight_bits", 16) or 16)
    weight_int_bits = int(notes.get("weight_int_bits", 6) or 6)
    bias_bits = int(notes.get("bias_bits", 24) or 24)
    bias_int_bits = int(notes.get("bias_int_bits", 10) or 10)
    accum_bits = int(notes.get("accum_bits", 24) or 24)
    accum_int_bits = int(notes.get("accum_int_bits", 10) or 10)

    conv_ic_unroll = 1
    conv_oc_unroll = 1
    dense_in_unroll = 1
    dense_out_unroll = 1
    partition_factor = int(notes.get("parallel_partition_factor", 1) or 1)
    pipeline_style = str(notes.get("parallel_pipeline_style", "balanced"))

    for lp in layer_plans:
        if lp.get("op_type") == "Conv":
            u = lp.get("unroll") or {}
            conv_ic_unroll = max(conv_ic_unroll, int(u.get("ic", 1) or 1))
            conv_oc_unroll = max(conv_oc_unroll, int(u.get("oc", 1) or 1))
        elif lp.get("op_type") == "Dense":
            u = lp.get("unroll") or {}
            dense_in_unroll = max(dense_in_unroll, int(u.get("in", 1) or 1))
            dense_out_unroll = max(dense_out_unroll, int(u.get("out", 1) or 1))

    if pipeline_style == "conservative":
        pipeline_ii = 2
    else:
        pipeline_ii = 1

    return f"""#pragma once
#include <ap_fixed.h>
#include <ap_int.h>

namespace fpgai {{

typedef ap_fixed<{act_bits}, {act_int_bits}, AP_TRN, AP_WRAP> act_t;
typedef ap_fixed<{weight_bits}, {weight_int_bits}, AP_TRN, AP_WRAP> wgt_t;
typedef ap_fixed<{bias_bits}, {bias_int_bits}, AP_TRN, AP_WRAP> bias_t;
typedef ap_fixed<{accum_bits}, {accum_int_bits}, AP_TRN, AP_WRAP> acc_t;
typedef ap_fixed<{accum_bits}, {accum_int_bits}, AP_TRN, AP_WRAP> accum_t;

#define FPGAI_PIPELINE_II {pipeline_ii}
#define FPGAI_PARTITION_FACTOR {max(1, partition_factor)}

#define FPGAI_CONV_IC_UNROLL {max(1, conv_ic_unroll)}
#define FPGAI_CONV_OC_UNROLL {max(1, conv_oc_unroll)}

#define FPGAI_DENSE_IN_UNROLL {max(1, dense_in_unroll)}
#define FPGAI_DENSE_OUT_UNROLL {max(1, dense_out_unroll)}

}} // namespace fpgai
"""