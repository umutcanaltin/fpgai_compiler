"""Single latency-only HLS reward: no area, power, energy or Pareto score."""
from math import isfinite, log


def qualified_latency_ns(cycles, requested_period_ns, *, feasible):
    if not feasible:
        return None
    if type(cycles) is not int or cycles <= 0:
        raise ValueError("Audited positive integer whole-top HLS cycles required")
    if type(requested_period_ns) not in (float, int) or (
            not isfinite(requested_period_ns) or requested_period_ns <= 0):
        raise ValueError("Audited positive requested HLS period required")
    return cycles * requested_period_ns


def latency_only_reward(previous_best_ns, updated_best_ns, *, qualified,
                        eval_penalty=0.005, infeasible_penalty=-0.05):
    if not qualified:
        return float(infeasible_penalty)
    if updated_best_ns is None or updated_best_ns <= 0 or not isfinite(updated_best_ns):
        raise ValueError("Qualified result needs an actual positive HLS latency")
    if previous_best_ns is None:
        return -eval_penalty
    if previous_best_ns <= 0 or not isfinite(previous_best_ns):
        raise ValueError("Invalid previous qualified latency")
    if updated_best_ns > previous_best_ns:
        raise ValueError("Best feasible latency must not regress")
    return log(previous_best_ns / updated_best_ns) - eval_penalty



def beginner_real_hls_latency_reward(*, qualified, hls_cycles, input_width,
                                     output_width, infeasible_penalty=-0.05):
    """Dense beginner reward from EVERY actual qualified HLS cycle count.

    For the one-operator MatMul task, MACs = input_width * output_width.
    r = MACs/(MACs + C_HLS) is strictly monotone decreasing in real qualified
    worst-case HLS cycles for a fixed task and fixed requested clock (32ns).
    The dimensions come from Functional Model IR, not from a QoR surrogate.
    All infeasible cases remain penalized; missing cycles are NEVER imputed.

    This intentionally differs from the older best-improvement-only reward;
    its scientific claim is training a policy for individually fast feasible
    architecture proposals, NOT mathematical equivalence to best-of-ten.
    """
    if not qualified:
        return float(infeasible_penalty)
    if (type(hls_cycles) is not int or hls_cycles <= 0
            or type(input_width) is not int or input_width <= 0
            or type(output_width) is not int or output_width <= 0):
        raise ValueError("Beginner reward requires positive source-attested cycles and exact MatMul dimensions")
    macs = input_width * output_width
    return float(macs / (macs + hls_cycles))



def curriculum_real_hls_cycle_reward(*, qualified, hls_cycles, model,
                                     requested_clock_ns):
    """Stage L0-L6: dense, strictly latency-monotonic native HLS reward.

    Sum the exact MatMul/Gemm MAC counts from the native Functional Model IR.
    Strictly feasible only when original CSim/HLS/source gates pass. The
    dimension-normalization is a task-size scale, not a synthetic HLS result.
    Normalize the actual qualified worst-cycle * requested-period nanoseconds
    by a fixed 32ns reference per MAC, so the reward also optimizes real HLS
    latency in the stages with a learnable requested clock. This is NOT a
    physical routed timing claim and never adds a second reward objective.
    """
    if not qualified:
        return -.05
    from math import isfinite
    if type(hls_cycles) is not int or hls_cycles <= 0:
        raise ValueError("Curriculum reward requires true bounded HLS worst-case cycles")
    if (type(requested_clock_ns) not in (float, int)
            or not isfinite(requested_clock_ns) or requested_clock_ns <= 0):
        raise ValueError("Curriculum reward requires original requested clock")
    macs = 0
    for op in model.operators:
        if op.kind not in ("MatMul", "Gemm"):
            raise ValueError("Unsupported research-stage operator")
        a = model.tensor(op.inputs[0]).shape
        w = model.tensor(op.inputs[1]).shape
        if len(a) != 2 or len(w) != 2 or a[1] != w[0]:
            raise ValueError("Invalid stage Functional Model IR MatMul dimensions")
        macs += a[0] * w[0] * w[1]
    if macs < 1:
        raise ValueError("No real stage MAC operations")
    reference_ns = macs * 32.
    actual_hls_ns = hls_cycles * float(requested_clock_ns)
    return float(reference_ns / (reference_ns + actual_hls_ns))
