"""Budgeted, evidence-separated search over real progressive Model IR contracts.

This is a *source-only research benchmark*. The oracle genuinely resolves,
lowers, verifies and emits the project's Model IR. Its two cost objectives are
DELIBERATELY hand-designed analytic proxies, NOT HLS estimates or FPGA QoR.
Tabular Q-learning is an experimental baseline, not a performance claim.
No vendor programs, trained models, network requests or file writes here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from math import ceil, prod
from random import Random
from typing import Callable
import json

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import ArchitectureError, TargetLimits
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement

SCHEMA = "fpgai-research/model-ir-source-only-search/v1"
AXES = ("precision_bits", "parallelism", "pes", "simd",
        "serialization", "tile", "pipeline_ii")
# Exact, preregistered finite action grid; 7 semantic axes, no hidden tuning.
OPTIONS = (
    (12, 16),              # Width only, fixed integer_bits=6; not accuracy-qualified
    (1, 2),                # Replicas
    (1, 2, 3),             # Processing elements
    (1, 2),                # SIMD
    (1, 2),                # Temporal reuse
    ((), (1, 2, 2)),      # Two 4x4-MatMul tile alternatives
    (None, 2),            # Pipeline request: absent or II=2; not achieved II
)
BASE = (1, 0, 0, 0, 0, 0, 0)  # width=16, otherwise minimal mapping
POLICIES = ("uniform", "random_walk", "q_learning",
            "random_global", "q_global")


@dataclass(frozen=True)
class Objective:
    source_signature: str
    cpp_sha256: str
    latency_proxy: int
    area_proxy: int
    score: float
    functional_sha256: str

    def as_dict(self):
        return asdict(self)


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _check_candidate(candidate, op_count):
    if len(candidate) != op_count or not all(
            len(row) == len(AXES) and all(type(index) is int
                and 0 <= index < len(OPTIONS[j])
                for j, index in enumerate(row))
            for row in candidate):
        raise ValueError("Search candidate is outside the preregistered Model IR grid")


def baseline(op_count: int):
    return tuple(BASE for _ in range(op_count))


def _requests(names, candidate):
    return {name: {axis: OPTIONS[j][indices[j]]
                   for j, axis in enumerate(AXES)}
            for name, indices in zip(names, candidate)}


def _resolve(model: FunctionalModel, candidate, max_parallel_units: int = 8):
    names = tuple(op.name for op in model.operators)
    _check_candidate(candidate, len(names))
    if any(op.kind not in ("MatMul", "Gemm") for op in model.operators):
        raise ValueError("Initial source-only search fixture supports only MatMul/Gemm")
    arch = resolve_scoped(
        model, {"precision_bits": 16, "integer_bits": 6},
        operators=_requests(names, candidate),
        limits=TargetLimits(max_parallel_units=max_parallel_units),
    )
    # The actual HLS emitter currently lacks explicit casts between mixed
    # numerical formats. Make precision a shared model-level mutation,
    # not a false legal operator-local transformation.
    if len({(p.effective.precision_bits, p.effective.integer_bits)
            for p in arch.points}) != 1:
        raise ArchitectureError("<model>", "MIXED_PRECISION",
                                "Emitter requires shared precision until cast lowering exists")
    return arch


def legal(model, candidate, *, max_parallel_units=8):
    """Fast legality oracle, shared by ALL local-neighborhood policies."""
    try:
        _resolve(model, candidate, max_parallel_units)
    except ArchitectureError:
        return False
    return True


def evaluate_source(model, candidate, *, max_parallel_units=8) -> Objective:
    """Actually resolve→lower→verify→hash Phi→emit HLS; *then* proxy costs.

    The proxy is an intentionally simplified structural scoring rule.
    It must never appear as Vitis cycles, routed LUTs, energy, or latency.
    """
    arch = _resolve(model, candidate, max_parallel_units)
    lowered = lower(arch)
    checked = verify_refinement(lowered)
    if not checked["stage_invariants_verified"]:
        raise AssertionError("Refinement invalid despite legal resolver result")
    signatures = [
        iteration_mapping_receipt(point, max_iterations=1000)["complete_mapping_sha256"]
        for point in arch.points
    ]
    emitted = emit_hls(lowered)
    latency = 0
    area = 0
    for point, op in zip(arch.points, lowered.operators):
        a = point.effective
        work = prod(point.iteration_shape)
        out_extent = point.iteration_shape[1]
        reduction_extent = point.iteration_shape[2]
        active_out = min(out_extent, a.parallelism * a.pes)
        active_reduce = min(reduction_extent, a.simd)
        tile_count = prod(ceil(d / t) for d, t in zip(
            point.iteration_shape, op.tile_shape))
        latency += (ceil(work / (active_out * active_reduce))
                    * a.serialization * (a.pipeline_ii or 1)
                    + 3 * tile_count)
        area += (a.precision_bits * a.parallelism * a.pes * a.simd
                 + 2 * prod(op.tile_shape) * a.precision_bits // 16)
    # Fixed normalizers derive from the baseline scale, not future observations.
    score = float(latency / 256.0 + area / 512.0)
    return Objective(
        source_signature=_digest(signatures),
        cpp_sha256=emitted.manifest["cpp_sha256"],
        latency_proxy=int(latency),
        area_proxy=int(area),
        score=score,
        functional_sha256=checked["functional_sha256"],
    )


def actions(op_count, *, global_only=False):
    # Precision is currently model-wide because no mixed-format cast lowering
    # exists. Other six dimensions can be operator-local in the scoped arm.
    return tuple(
        (op, axis, delta)
        for axis, options in enumerate(OPTIONS)
        for op in ((None,) if global_only or axis == 0 else range(op_count))
        if len(options) > 1
        for delta in (-1, 1)
    )


def apply(candidate, action):
    op, axis, delta = action
    indexes = [list(row) for row in candidate]
    selected = range(len(indexes)) if op is None else (op,)
    if op is None and len({row[axis] for row in indexes}) != 1:
        raise ValueError("Global action requires an initially global candidate")
    next_value = indexes[0 if op is None else op][axis] + delta
    if not 0 <= next_value < len(OPTIONS[axis]):
        return None
    for index in selected:
        indexes[index][axis] = next_value
    return tuple(tuple(row) for row in indexes)


def _argmin(items, rng):
    low = min(value for _, value in items)
    tied = [key for key, value in items if value == low]
    return rng.choice(tied)


def _choose_q(rng, q_table, state, choices, epsilon):
    if rng.random() < epsilon:
        return rng.choice(choices)
    values = [(item, q_table.get((state, item), 0.0)) for item in choices]
    high = max(value for _, value in values)
    return rng.choice([item for item, value in values if value == high])


def run_search(model: FunctionalModel,
               method: str, seed: int, budget: int = 30,
               *, max_parallel_units=8,
               evaluator: Callable | None = None):
    """Identical costed evaluations for all algorithms; Q-learning is online.

    Search budget counts unique, genuinely evaluated legal candidates,
    INCLUDING the common baseline. Cheap IR legality checks and repeated
    proposals are counted separately and never recharged as oracle calls.
    """
    if (method not in POLICIES or type(seed) is not int
            or type(budget) is not int or not 2 <= budget <= 500
            or not 0 <= seed < 2**32):
        raise ValueError("Invalid method/seed/budget")
    if type(max_parallel_units) is not int or max_parallel_units < 1:
        raise ValueError("Invalid target structural-capacity proxy")
    rng = Random(seed)
    names = tuple(op.name for op in model.operators)
    if not names or len(names) > 4:
        raise ValueError("Search study limited to 1..4 named operators")
    current = baseline(len(names))
    oracle = evaluator or evaluate_source
    def measure(key):
        return oracle(model, key, max_parallel_units=max_parallel_units)
    first = measure(current)
    fixed_hash = functional_fingerprint(model)
    if first.functional_sha256 != fixed_hash:
        raise ValueError("Invalid search oracle: functional graph identity changed")
    visited = {current: first}
    seen_source = {first.source_signature}
    best = current
    state_score = first.score
    best_score = state_score
    q_table = {}
    global_only = method in ("random_global", "q_global")
    moves = actions(len(names), global_only=global_only)
    attempts = invalid = duplicated = source_duplicates = 0
    improvements = 0
    trace = [{"evaluation": 1, "best_score": best_score,
              "latency_proxy": first.latency_proxy,
              "area_proxy": first.area_proxy}]
    # Bound proposal overhead even when search cannot produce a new legal point.
    max_attempts = budget * 100
    while len(visited) < budget and attempts < max_attempts:
        attempts += 1
        if method == "uniform":
            candidate = tuple(tuple(rng.randrange(len(opts)) for opts in OPTIONS)
                              for _ in names)
            action = None
            state = None
        else:
            # Same restart and neighborhood for random walk and Q-learning.
            if attempts % 11 == 0:
                current = best
                state_score = visited[current].score
            state = (int(attempts > max_attempts // 2),
                     int(state_score > best_score + 1e-12))
            valid_moves = [action for action in moves
                           if (apply(current, action) is not None)]
            if method in ("random_walk", "random_global"):
                action = rng.choice(valid_moves)
            else:
                action = _choose_q(rng, q_table, state, valid_moves, epsilon=0.25)
            candidate = apply(current, action)
        if candidate is None or not legal(model, candidate,
                                         max_parallel_units=max_parallel_units):
            invalid += 1
            reward = -0.05
            following = current
        elif candidate in visited:
            duplicated += 1
            reward = -0.02
            following = current
        else:
            result = measure(candidate)
            if result.functional_sha256 != fixed_hash:
                raise ValueError("Search mutated the frozen mathematical model")
            visited[candidate] = result
            prev = state_score
            state_score = result.score
            reward = max(-1.0, min(1.0, (prev - state_score) / max(abs(prev), 1e-9)))
            following = candidate
            if result.source_signature in seen_source:
                source_duplicates += 1
            else:
                seen_source.add(result.source_signature)
            if result.score < best_score - 1e-12:
                best_score = result.score
                best = candidate
                reward += 0.2
                improvements += 1
            trace.append({"evaluation": len(visited), "best_score": best_score,
                          "latency_proxy": visited[best].latency_proxy,
                          "area_proxy": visited[best].area_proxy})
        if method != "uniform":
            if method in ("q_learning", "q_global"):
                next_state = (int(attempts > max_attempts // 2),
                              int(state_score > best_score + 1e-12))
                # Tabular off-policy temporal difference update (Q-learning).
                bootstrap = max(q_table.get((next_state, a), 0.0)
                                for a in moves)
                old = q_table.get((state, action), 0.0)
                q_table[(state, action)] = old + 0.35 * (
                    reward + 0.85 * bootstrap - old)
            current = following
    if len(visited) != budget:
        raise RuntimeError("Search exhausted bounded proposals before evaluation budget")
    best_observation = visited[best]
    # Distinct observed IR mappings, NOT necessarily distinct routed circuits.
    return {
        "method": method,
        "search_representation": "global" if global_only else "operator_scoped",
        "seed": seed, "evaluation_budget": budget,
        "evaluations": len(visited), "proposals": attempts,
        "invalid_proposals": invalid, "repeated_proposals": duplicated,
        "distinct_source_mapping_observations": len(seen_source),
        "source_signature_collisions": source_duplicates,
        "best_candidate": [list(row) for row in best],
        "best": best_observation.as_dict(),
        "improvements": improvements,
        "trace": trace,
        "trained_q_entries": len(q_table),
        "evidence_level": "E0/E1-and-analytic-PROXY-only",
        "native_vendor_tools_executed": False,
    }


def benchmark(model, seeds=(3, 5, 7, 11), budget=30):
    """Frozen paired seeds, same legal oracle, source evaluator and budget."""
    seeds = tuple(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and distinct")
    rows = [run_search(model, method, seed, budget)
            for seed in seeds for method in POLICIES]
    return {
        "schema": SCHEMA,
        "workload_functional_sha256": functional_fingerprint(model),
        "fixture_operators": [op.name for op in model.operators],
        "axes": list(AXES),
        "option_values": [
            [list(v) if isinstance(v, tuple) else v for v in options]
            for options in OPTIONS
        ],
        "methods": list(POLICIES), "seeds": list(seeds),
        "budget_per_method_per_seed": budget,
        "rows": rows,
        "objective_definition": (
            "Two deterministic hand-designed source-only structural "
            "proxies. No Vitis measurements, no calibrated performance "
            "prediction, no fixed-point correctness qualification."
        ),
        "source_evidence_only": True,
        "native_vendor_tools_executed": False,
        "search_superiority_claimed": False,
        "comparison_scope": (
            "q_learning versus random_walk isolates the policy on the same "
            "operator-scoped action space; q_global versus random_global "
            "isolates policy on the global-only action space. Global versus "
            "scoped comparisons change representational expressiveness."
        ),
    }
