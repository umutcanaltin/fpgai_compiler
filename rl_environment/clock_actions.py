"""Bounded learnable floating-point HLS clock actions, not a clock grid.

The shared RL policy outputs a floating requested period with 1 ps resolution.
Bounds are an internal safety/experiment envelope, not FPGA achievable Fmax.
The explicit, optional fixed-clock arm exists ONLY for registered ablations.
"""
from __future__ import annotations

from math import isfinite
from numbers import Real

CLOCK_MIN_NS = 2.0
CLOCK_MAX_NS = 32.0
CLOCK_DECIMALS = 3
CLOCK_RESOLUTION_NS = 0.001
CLOCK_ACTION_MODE = "learned_beta_float_period_ns_v1"


def normalize_clock_period_ns(period, *, decimals=CLOCK_DECIMALS):
    """Map a learned floating proposal to an exact reproducible Tcl period.

    Vitis Tcl is emitted using significant digits; 3 decimal places avoid
    action/report mismatch and provide sub-nanosecond search resolution.
    No unknown HLS feasibility is inferred by this normalization.
    """
    # Gymnasium Box(shape=()) may pass a NumPy scalar or 0-D array.
    if getattr(period, "shape", None) == () and hasattr(period, "item"):
        period = period.item()
    if isinstance(period, bool) or not isinstance(period, Real) or not isfinite(period):
        raise ValueError("Clock action must be finite floating-point ns")
    if not CLOCK_MIN_NS <= period <= CLOCK_MAX_NS:
        raise ValueError("Clock action outside internal safety bounds")
    if decimals not in (2, 3):
        raise ValueError("Supported original-Vitis clock precision is 0.01 or 0.001 ns")
    normalized = round(float(period), decimals)
    if not CLOCK_MIN_NS <= normalized <= CLOCK_MAX_NS:
        raise ValueError("Normalized requested clock outside safety bounds")
    return normalized


def validate_clock_action(task, period):
    """Strictly validate a policy-chosen, canonical requested clock."""
    # Vitis HLS 2023.2 reports TargetClockPeriod to 0.01 ns in the
    # observed original source-pinned XML. New L5/L6 stages therefore
    # submit 0.01 ns requested clocks; historical Gym remains 0.001 ns.
    decimals = 2 if task.curriculum_stage in ("L5", "L6") else CLOCK_DECIMALS
    normalized = normalize_clock_period_ns(period, decimals=decimals)
    if abs(period - normalized) > 1e-8:
        raise ValueError(f"Clock action must be canonical to {10 ** -decimals:.3f} ns")
    if task.clock_ns is None:
        fixed = task.fixed_clock_ablation_ns
        if fixed is not None and normalized != fixed:
            raise ValueError("Clock action differs from fixed-clock ablation")
    elif normalized not in task.clock_choices_ns:
        raise ValueError("Clock period is outside legacy task legal grid")
    return normalized


def action_bounds_ns(task):
    fixed = task.fixed_clock_ablation_ns if task.clock_ns is None else task.clock_ns
    if fixed is not None:
        return float(fixed), float(fixed)
    return CLOCK_MIN_NS, CLOCK_MAX_NS
