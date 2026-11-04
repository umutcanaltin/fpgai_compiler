"""Create a seven-axis research *experiment plan* from real Model IR geometry.

This is not automatic architecture optimization. The researcher explicitly
selects an operator or accepts the first compute operator, and every generated
variant is subject to the existing fail-closed architecture resolver.
"""
from __future__ import annotations
from fpgai_research.architecture.resolver import iteration_shape
from fpgai_research.experiments.full_pipeline import SCHEMA, parse_plan

def make_seven_axis_plan(model, operator=None, *, precision_bits=16,
                         integer_bits=6,part="xck26-sfvc784-2LV-c",clock_ns=5.0):
    model.validate()
    compute=[op for op in model.operators if op.kind in ("MatMul","Gemm","Conv")]
    if operator is None:
        if not compute:
            raise ValueError("No HLS compute operator eligible for seven-axis study")
        operator=compute[0].name
    selected=next((op for op in compute if op.name==operator),None)
    if selected is None:
        raise ValueError("Seven-axis plan requires a named MatMul, Gemm or Conv operator")
    dims=iteration_shape(model,selected)
    if selected.kind in ("MatMul","Gemm"):
        m,n,k=dims
        tile=[m,max(1,n//2),max(1,k//2)]
    else:
        batch,cout,height,width,cin,kh,kw=dims
        tile=[batch,max(1,cout//2),max(1,height//2),max(1,width//2),cin,kh,kw]
    proposal={"schema":SCHEMA,"operator":operator,
       "default":{"precision_bits":precision_bits,"integer_bits":integer_bits,
          "parallelism":1,"pes":1,"simd":1,"serialization":1,
          "tile":[],"pipeline_ii":None},
       "variants":{"precision_bits":[x for x in (8,12,16,32) if x!=precision_bits],
          "parallelism":[2],"pes":[2],"simd":[2],
          "serialization":[2],"tile":[tile],"pipeline_ii":[2]},
       "interactions":[{"parallelism":2,"pes":2,"simd":2,"serialization":2}],
       "target_part":part,"clock_ns":clock_ns,"tolerance":0.1}
    parse_plan(proposal)
    return proposal



def make_comprehensive_seven_axis_plan(model, operator=None, *,
                                       part="xck26-sfvc784-2LV-c",
                                       clock_ns=5.0):
    """Preregister main effects and all 21 pairwise seven-axis interactions.

    This is a bounded, non-Cartesian set of source-level requests.
    Invalid configurations remain explicit blockers; no physical results
    or valid native schedules are inferred from the proposed points.
    """
    from itertools import combinations

    plan = make_seven_axis_plan(
        model, operator=operator, part=part, clock_ns=clock_ns
    )
    # Match the separately preregistered native output maximum-error
    # acceptance threshold. The compiler's host-float check is still
    # weaker than a source-attested fixed-point C-simulation receipt.
    plan["tolerance"] = 0.05
    base_tile = plan["variants"]["tile"][0]
    second_tile = [
        1 if extent == 1 else max(1, extent // 2)
        for extent in base_tile
    ]
    tile_options = [base_tile]
    if second_tile != base_tile:
        tile_options.append(second_tile)
    plan["variants"] = {
        "precision_bits": [8, 12, 32],
        "parallelism": [2, 4],
        "pes": [2, 4],
        "simd": [2, 4],
        "serialization": [2, 4, 8],
        "tile": tile_options,
        "pipeline_ii": [1, 2, 4],
    }
    from fpgai_research.experiments.campaign import DIMENSIONS
    changes = {name: plan["variants"][name][0] for name in DIMENSIONS}
    plan["interactions"] = [
        {a: changes[a], b: changes[b]}
        for a, b in combinations(DIMENSIONS, 2)
    ]
    # Additional three-/four-dimensional interaction contrasts examine
    # commonly coupled arithmetic and scheduling variables, without
    # generating the enormous (and often infeasible) Cartesian product.
    plan["interactions"].extend([
        {"pes": 2, "simd": 2, "serialization": 2},
        {"parallelism": 2, "tile": tile_options[0], "pipeline_ii": 1},
        {"precision_bits": 8, "simd": 2, "serialization": 2, "pipeline_ii": 1},
    ])
    parse_plan(plan)
    return plan
