"""Lower resolved architecture into explicit HLS mechanism contracts."""
from fpgai_research.model_ir.contracts import ArchitecturalModel, LoweredModel, LoweredOperator

def lower(model: ArchitecturalModel) -> LoweredModel:
    operators = {op.name: op for op in model.functional.operators}
    points = {p.operator: p for p in model.points}
    if len(points) != len(model.points) or set(operators) != set(points):
        raise ValueError("Resolution must cover each operator exactly once")
    lowered = []
    for name in (op.name for op in model.functional.operators):
        p = points[name]
        a = p.effective
        if a != p.requested:
            raise ValueError("Research resolver cannot silently substitute requested architecture")
        lowered.append(LoweredOperator(
            name=name, kind=operators[name].kind, iteration_shape=p.iteration_shape,
            architecture=a, source_scope=p.source_scope, tile_shape=a.tile or p.iteration_shape,
            replica_count=a.parallelism, pe_count=a.pes, simd_lanes=a.simd,
            fold_factor=a.serialization, pipeline_ii=a.pipeline_ii,
            tail_policy="bounds-guard",
            numerical_type=f"ap_fixed<{a.precision_bits},{a.integer_bits},AP_RND,AP_SAT>",
            decision_sources=p.decision_sources,
        ))
    return LoweredModel(model, tuple(lowered))
