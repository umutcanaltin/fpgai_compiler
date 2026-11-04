"""Policy-neutral, source-backed RL action space for *implemented* FPGAI knobs.

This is the compiler-facing RL planning contract, not a trained RL agent or
a synthetic HLS environment. Actions reuse resolve_scoped ->
compile_architecture; there is exactly one authoritative source emitter.
Every accepted action denotes a *complete* compiler source state. It does NOT
assert that the part will fit, Vitis will meet timing, or a board will run.

The legacy MatMul/Gemm native Gym and its historical task/reward hashes are
unchanged. This new source-only interface also admits supported Conv networks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from math import isfinite, prod
import re
from typing import Any, Iterable

from fpgai_research.architecture.resolver import ArchitectureError, TargetLimits, iteration_shape
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.compiler import compile_architecture
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel
from fpgai_research.model_ir.semantics import functional_fingerprint
from .clock_actions import CLOCK_MAX_NS, CLOCK_MIN_NS, normalize_clock_period_ns

SCHEMA = "fpgai-research/rl-source-hardware-space/v1"
ARCHITECTURE_FIELDS = tuple(f.name for f in fields(ArchitectureRequest))
NUMERIC_FIELDS = ("precision_bits", "integer_bits")
OPERATOR_FIELDS = tuple(x for x in ARCHITECTURE_FIELDS if x not in NUMERIC_FIELDS)
COMPUTE_KINDS = frozenset(("MatMul", "Gemm", "Conv"))
# These are deliberately NOT pretend supported RL actions. Some hardware
# library/system compiler features exist elsewhere but have no tested
# architecture-action-to-source-and-native HLS adapter in this research Gym.
RESOURCE_NAMES = ("lut", "ff", "dsp", "bram_18k", "uram")
FUTURE_FIELDS = (
    "tensor_exchange_policy", "memory_banking", "memory_port_count", "bram_uram_placement",
    "dma_width_bits", "dma_burst_length", "axi_interconnect",
    "clock_domain", "clock_crossing", "device_part",
    "physical_placement", "routing_strategy", "power_target",
)


def _sha(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _typesafe_tuple(value: object) -> tuple:
    if not isinstance(value, (tuple, list)) or any(type(x) is not int for x in value):
        raise TypeError("Tile action expects a vector of integer iteration extents")
    return tuple(value)


@dataclass(frozen=True)
class HardwareState:
    """A full network configuration with global numeric format and clock."""
    model: FunctionalModel
    requests: tuple[ArchitectureRequest, ...]
    part: str
    clock_ns: float
    limits: TargetLimits = TargetLimits(max_parallel_units=8)
    resource_caps: tuple[int, int, int, int, int] | None = None
    max_absolute_error: float | None = None

    def __post_init__(self):
        if not isinstance(self.model, FunctionalModel):
            raise TypeError("Expected original FunctionalModel")
        self.model.validate()
        if not self.model.operators or len(self.requests) != len(self.model.operators):
            raise ValueError("One architecture request per operator required")
        if type(self.requests) is not tuple or any(
            not isinstance(req, ArchitectureRequest) for req in self.requests
        ):
            raise TypeError("Complete immutable architecture requests required")
        for field in NUMERIC_FIELDS:
            if len({getattr(req, field) for req in self.requests}) != 1:
                raise ValueError("Mixed operator numeric formats require explicit casts")
        if not isinstance(self.limits, TargetLimits):
            raise TypeError("Compiler target limits required")
        if not isinstance(self.part, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{3,100}", self.part, re.ASCII
        ):
            raise ValueError("A safe explicit hardware part is required")
        if isinstance(self.clock_ns, bool) or type(self.clock_ns) not in (int, float):
            raise ValueError("Clock period must be a real, canonical number")
        rounded = normalize_clock_period_ns(self.clock_ns)
        if self.clock_ns != rounded:
            raise ValueError("Clock period must be canonical to 0.001 ns")
        if self.resource_caps is not None and (
            type(self.resource_caps) is not tuple or len(self.resource_caps) != 5
            or any(type(v) is not int or v < 0 for v in self.resource_caps)
        ):
            raise ValueError("Exactly five nonnegative original-HLS resource caps required")
        if self.max_absolute_error is not None and (
            type(self.max_absolute_error) not in (float, int)
            or not isfinite(self.max_absolute_error)
            or self.max_absolute_error <= 0
        ):
            raise ValueError("Numerical absolute-error limit must be finite and positive")

    @property
    def functional_sha256(self) -> str:
        return functional_fingerprint(self.model)

    @property
    def state_sha256(self) -> str:
        return _sha({
            "schema": SCHEMA,
            "functional_sha256": self.functional_sha256,
            "part": self.part,
            "clock_ns": self.clock_ns,
            "limits": asdict(self.limits),
            "resource_caps": self.resource_caps,
            "max_absolute_error": self.max_absolute_error,
            "requests": [asdict(req) for req in self.requests],
        })


@dataclass(frozen=True)
class HardwareEdit:
    """One discrete architecture edit, or a continuous global clock edit."""
    scope: str  # numeric, operator, clock
    field: str
    value: object
    operator_index: int | None = None


@dataclass(frozen=True)
class HardwareTransition:
    status: str  # SOURCE_ACCEPTED, REJECTED
    previous: HardwareState
    next: HardwareState
    action: HardwareEdit
    reason: str | None = None
    source_cpp_sha256: str | None = None

    def receipt(self) -> dict:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "previous_state_sha256": self.previous.state_sha256,
            "next_state_sha256": (
                self.next.state_sha256 if self.status == "SOURCE_ACCEPTED" else None
            ),
            "functional_sha256": self.previous.functional_sha256,
            "source_cpp_sha256": self.source_cpp_sha256,
            "action": asdict(self.action),
            "reason": self.reason,
            "evidence": "E0-refinement-E1-source-emitted" if self.source_cpp_sha256
                        else "SOURCE_REJECTED",
            "native_vitis_calls": 0,
            "native_vivado_calls": 0,
            "hls_latency_ns": None,
            "reward": None,
            "hardware_feasibility": "UNKNOWN",
            "physical_board_inference_verified": False,
        }


def initial_state(model: FunctionalModel, *, part: str,
                  clock_ns: float = 10.0,
                  limits: TargetLimits = TargetLimits(max_parallel_units=8),
                  resource_caps: tuple[int, int, int, int, int] | None = None,
                  max_absolute_error: float | None = None) -> HardwareState:
    """Start from the exact currently supported compiler request defaults."""
    state = HardwareState(
        model, tuple(ArchitectureRequest() for _ in model.operators),
        part, clock_ns, limits, resource_caps, max_absolute_error,
    )
    validate_source(state)
    return state


def compile_state(state: HardwareState):
    """Only original compiler resolver/IR/refinement/emitter; no vendor execution."""
    arch = resolve_scoped(
        state.model, ArchitectureRequest(),
        operators={
            op.name: asdict(req)
            for op, req in zip(state.model.operators, state.requests)
        },
        limits=state.limits,
    )
    # The HLS emitter today uses ONE ap_fixed data_t throughout the model.
    # The globally synchronized numeric request is an actual hard condition.
    artifact = compile_architecture(arch)
    expected = state.functional_sha256
    if (artifact.refinement_receipt.get("stage_invariants_verified") is not True
            or artifact.refinement_receipt.get("functional_sha256") != expected
            or artifact.emission.manifest.get("functional_sha256") != expected):
        raise ValueError("Original compiler Model IR/source refinement failed")
    cpp = artifact.emission.cpp
    digest = sha256(cpp.encode("utf-8")).hexdigest()
    if artifact.emission.manifest.get("cpp_sha256") != digest:
        raise ValueError("Generated source does not match compiler manifest")
    return artifact


def validate_source(state: HardwareState) -> dict:
    """Compile and attest a complete RL architecture state, source evidence only."""
    artifact = compile_state(state)
    digest = sha256(artifact.emission.cpp.encode("utf-8")).hexdigest()
    return {
        "schema": SCHEMA,
        "state_sha256": state.state_sha256,
        "functional_sha256": state.functional_sha256,
        "cpp_sha256": digest,
        "operator_count": len(artifact.lowered.architectural.points),
        "source_compilable": True,
        "source_evidence_only": True,
        "native_vitis_calls": 0,
        "native_vivado_calls": 0,
        "hardware_qor_known": False,
    }


def native_evaluation_plan(state: HardwareState) -> dict:
    """Exact compiler-to-real-HLS handoff *plan*, never a native execution.

    A real CSim/CSynth evaluator still needs original fixed numerical samples,
    an immutable per-candidate output directory, verified tool invocation
    and independent receipt audit. Do not return a synthetic reward here.
    """
    if state.resource_caps is None or state.max_absolute_error is None:
        raise ValueError(
            "Real HLS evaluation requires all five explicit resource caps "
            "and a positive numerical absolute-error tolerance"
        )
    compiled = validate_source(state)
    return {
        "schema": "fpgai-research/rl-native-hls-evaluation-plan/v1",
        "status": "SOURCE_ONLY_NATIVE_EVALUATION_NOT_EXECUTED",
        "state_sha256": state.state_sha256,
        "functional_sha256": state.functional_sha256,
        "cpp_sha256": compiled["cpp_sha256"],
        "part": state.part,
        "requested_clock_ns": state.clock_ns,
        "resource_caps": dict(zip(RESOURCE_NAMES, state.resource_caps)),
        "max_absolute_error": state.max_absolute_error,
        "complete_architecture": [
            {"operator_index": i, "kind": op.kind, "request": asdict(req)}
            for i, (op, req) in enumerate(zip(state.model.operators, state.requests))
        ],
        "requires_source_attested_original_vitis_csim_and_csynth": True,
        "requires_fixed_numerical_samples_and_testbench": True,
        "requires_effective_hls_timing_qualification": True,
        "qualified_latency_formula": "whole_top_worst_cycles * requested_clock_ns",
        "actual_native_hls_latency_ns": None,
        "actual_resource_occupancy": None,
        "reward": None,
        "native_vitis_calls": 0,
        "physical_timing_achieved": False,
    }


def _value_candidates(value: int, *, max_value: int = 32) -> tuple[int, ...]:
    """Finite research menu, not an exhaustive legal integer domain."""
    return tuple(x for x in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
                 if x <= max_value or x == value)


def _tile_domain(kind: str, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Only source-implemented tile axes; bounds derived from the real loop."""
    tiles = [()]
    if kind in ("MatMul", "Gemm"):
        axes = (0, 1, 2)
    elif kind == "Conv":
        axes = (1, 2, 3)
    else:
        return ((),)
    full = tuple(shape)
    tiles.append(full)
    for axis in axes:
        for extent in (1, 2, 4, 8, 16, 32):
            if extent <= shape[axis]:
                candidate = list(full)
                candidate[axis] = extent
                tiles.append(tuple(candidate))
    return tuple(dict.fromkeys(tiles))


def _source_domains(state: HardwareState, index: int) -> dict[str, tuple]:
    op = state.model.operators[index]
    req = state.requests[index]
    shape = iteration_shape(state.model, op)
    if op.kind not in COMPUTE_KINDS:
        return {}
    cap = state.limits.max_parallel_units or 32
    return {
        "parallelism": _value_candidates(req.parallelism, max_value=cap),
        "pes": _value_candidates(req.pes, max_value=cap),
        "simd": _value_candidates(req.simd, max_value=cap),
        "serialization": _value_candidates(req.serialization),
        "tile": _tile_domain(op.kind, tuple(shape)),
        "pipeline_ii": (None, 1, 2, 4, 8),
    }


def knob_inventory(state: HardwareState) -> dict:
    """All implemented request *categories*, plus explicit unavailable knobs.

    A domain is a bounded proposal menu, never a promise of full legal
    combination, source behavioural effect, or physical timing feasibility.
    """
    rows = []
    for index, (op, req) in enumerate(zip(state.model.operators, state.requests)):
        domains = _source_domains(state, index)
        rows.append({
            "operator_index": index,
            "kind": op.kind,
            "iteration_shape": list(iteration_shape(state.model, op)),
            "active_operator_knobs": {
                field: [list(v) if isinstance(v, tuple) else v for v in values]
                for field, values in domains.items()
            },
            "fixed_operator_knobs": {
                field: getattr(req, field)
                for field in OPERATOR_FIELDS if field not in domains
            },
            "current": asdict(req),
        })
    current = state.requests[0]
    return {
        "schema": SCHEMA,
        "model_global_numeric": {
            "precision_bits": list(state.limits.supported_widths),
            "integer_bits": list(range(1, current.precision_bits + 1)),
            "current": {f: getattr(current, f) for f in NUMERIC_FIELDS},
            "reason": "One global ap_fixed format; no cross-operator cast lowering",
        },
        "global_continuous_clock": {
            "field": "clock_ns",
            "min_ns": CLOCK_MIN_NS, "max_ns": CLOCK_MAX_NS,
            "resolution_ns": 0.001,
            "current_ns": state.clock_ns,
            "meaning": "requested Vitis clock, NOT measured FPGA Fmax",
        },
        "operators": rows,
        "declared_not_yet_actionable": [
            {"field": field, "status": "UNIMPLEMENTED_IN_THIS_ACTION_BACKEND"}
            for field in FUTURE_FIELDS
        ],
        "part": state.part,
        "resource_caps": (dict(zip(RESOURCE_NAMES, state.resource_caps))
                          if state.resource_caps is not None else None),
        "max_absolute_error": state.max_absolute_error,
        "resource_caps_are_native_task_constraints_not_cost_predictions": True,
        "part_is_fixed_task_context_not_rl_action": True,
        "candidate_domains_exhaustive": False,
        "native_tool_calls": 0,
    }


def _patch(state: HardwareState, action: HardwareEdit) -> HardwareState:
    if not isinstance(action, HardwareEdit):
        raise TypeError("Hardware action must be a typed HardwareEdit")
    reqs = list(state.requests)
    if action.scope == "clock":
        if action.operator_index is not None or action.field != "clock_ns":
            raise ValueError("Clock is a global requested-period action only")
        value = normalize_clock_period_ns(action.value)
        if value != action.value:
            raise ValueError("Requested clock must be canonical to 0.001 ns")
        return replace(state, clock_ns=value)
    if action.scope == "numeric":
        if action.operator_index is not None or action.field not in NUMERIC_FIELDS:
            raise ValueError("Numeric format must be a globally synchronized action")
        if type(action.value) is not int:
            raise TypeError("Numeric action value must be an integer")
        return replace(state, requests=tuple(
            replace(req, **{action.field: action.value}) for req in reqs
        ))
    if action.scope != "operator" or action.field not in OPERATOR_FIELDS:
        raise ValueError("Unimplemented/non-compiler knob cannot be proposed as an action")
    i = action.operator_index
    if type(i) is not int or not 0 <= i < len(reqs):
        raise ValueError("Operator index is missing or outside model")
    if action.field not in _source_domains(state, i):
        raise ValueError("This operator has no active source-backed version of the knob")
    if action.field == "tile":
        value = _typesafe_tuple(action.value)
    else:
        value = action.value
        if value is not None and type(value) is not int:
            raise TypeError("Hardware mapping value must be integer or null for II")
    # Bound the *RL* proposal domain even when the underlying compiler
    # accepts arbitrary positive integers. A large serialization/tiling
    # request must not silently slip past the declared exploration budget.
    # This does NOT replace the later original compiler legality check.
    if value not in _source_domains(state, i)[action.field]:
        raise ValueError("Action value is outside this bounded hardware knob menu")
    reqs[i] = replace(reqs[i], **{action.field: value})
    return replace(state, requests=tuple(reqs))


def transition(state: HardwareState, action: HardwareEdit) -> HardwareTransition:
    """Source-compile exact next full design or fail without changing state."""
    try:
        next_state = _patch(state, action)
        if next_state.state_sha256 == state.state_sha256:
            raise ValueError("NO_CHANGE: action does not modify source design")
        emitted = validate_source(next_state)
    except (ValueError, TypeError, OverflowError, ArchitectureError, IndexError) as exc:
        return HardwareTransition("REJECTED", state, state, action, str(exc))
    return HardwareTransition(
        "SOURCE_ACCEPTED", state, next_state, action,
        source_cpp_sha256=emitted["cpp_sha256"],
    )


def candidate_actions(state: HardwareState, *, max_actions: int = 512
                      ) -> tuple[HardwareEdit, ...]:
    """Deterministic, complete discrete proposal menu for a *bounded* catalog.

    Continuous clock_ns is emitted by a separate real-valued policy head.
    No source or native tool is invoked when enumerating candidate actions.
    Never silently truncate: the caller must explicitly adjust the budget.
    """
    if type(max_actions) is not int or not 1 <= max_actions <= 512:
        raise ValueError("Finite explicit candidate menu budget required")
    current = state.requests[0]
    edits = [
        HardwareEdit("numeric", "precision_bits", width)
        for width in state.limits.supported_widths
        if width != current.precision_bits
    ]
    edits += [
        HardwareEdit("numeric", "integer_bits", integer)
        for integer in range(1, current.precision_bits + 1)
        if integer != current.integer_bits
    ]
    for i, req in enumerate(state.requests):
        for field, domain in _source_domains(state, i).items():
            for value in domain:
                if value != getattr(req, field):
                    edits.append(HardwareEdit("operator", field, value, i))
    if len(edits) > max_actions:
        raise ValueError(
            f"Candidate action catalog has {len(edits)} values, exceeds "
            f"explicit max_actions={max_actions}; refusing silent truncation"
        )
    return tuple(edits)


def action_features(state: HardwareState, actions: tuple[HardwareEdit, ...]) -> list[dict]:
    """Policy-ready, variable-length typed action descriptions without rewards.

    Keep explicit category/value information; no fictitious resource cost,
    gradient, value estimate or vendor timing is attached to source actions.
    """
    if not isinstance(actions, tuple) or any(
        not isinstance(action, HardwareEdit) for action in actions
    ):
        raise TypeError("Typed immutable action candidates required")
    rows = []
    for action in actions:
        # Confirm this is a syntactically meaningful editable field/target.
        # Full legality still belongs to the actual compiler transition.
        next_state = _patch(state, action)
        rows.append({
            "scope": action.scope,
            "operator_index": action.operator_index,
            "field": action.field,
            "field_index": (
                ARCHITECTURE_FIELDS.index(action.field)
                if action.field in ARCHITECTURE_FIELDS else len(ARCHITECTURE_FIELDS)
            ),
            "value": list(action.value) if isinstance(action.value, tuple)
                     else action.value,
            "source_change": next_state.state_sha256 != state.state_sha256,
            "native_qor": None,
        })
    return rows


def action_mask(state: HardwareState, actions: tuple[HardwareEdit, ...]) -> dict:
    """Source-backed finite masks; no fabricated HLS, resources or reward."""
    if type(actions) is not tuple or len(actions) > 512 or any(
        not isinstance(a, HardwareEdit) for a in actions
    ):
        raise ValueError("Provide <=512 typed actions to evaluate exactly")
    decisions = [transition(state, action) for action in actions]
    return {
        "schema": SCHEMA,
        "state_sha256": state.state_sha256,
        "source_compilable": [d.status == "SOURCE_ACCEPTED" for d in decisions],
        "reason": [d.reason for d in decisions],
        "source_cpp_sha256": [d.source_cpp_sha256 for d in decisions],
        "action_count": len(decisions),
        "source_compilability_not_native_feasibility": True,
        "native_vitis_calls": 0,
        "actual_hls_resource_estimates": None,
        "actual_hls_latency_ns": None,
        "reward": None,
    }


def graph_observation(state: HardwareState) -> dict:
    """Variable-sized public graph without tensor values or private weights."""
    tensor_order = tuple(t.name for t in state.model.tensors)
    tensors = {name: i for i, name in enumerate(tensor_order)}
    op_nodes = {op.name: len(tensor_order) + i for i, op in enumerate(state.model.operators)}
    nodes = []
    for t in state.model.tensors:
        nodes.append({
            "id": tensors[t.name], "type": "tensor",
            "rank": len(t.shape), "shape": list(t.shape),
            "constant": t.data is not None, "element_count": prod(t.shape),
        })
    edges = []
    for i, (op, request) in enumerate(zip(state.model.operators, state.requests)):
        oid = op_nodes[op.name]
        nodes.append({
            "id": oid, "type": "operator", "kind": op.kind,
            "iteration_shape": list(iteration_shape(state.model, op)),
            "mapping": asdict(request),
        })
        for position, name in enumerate(op.inputs):
            edges.append({"src": tensors[name], "dst": oid,
                          "type": "consumes", "port": position})
        for name in op.outputs:
            edges.append({"src": oid, "dst": tensors[name],
                          "type": "produces"})
    return {
        "schema": SCHEMA,
        "observation_type": "typed-variable-size-graph-without-weights",
        "nodes": nodes, "edges": edges,
        "global": {
            "requested_clock_ns": state.clock_ns,
            "target_max_parallel_units": state.limits.max_parallel_units,
            "target_supported_widths": list(state.limits.supported_widths),
            "operator_count": len(state.model.operators),
            "resource_caps": (dict(zip(RESOURCE_NAMES, state.resource_caps))
                              if state.resource_caps is not None else None),
            "max_absolute_error": state.max_absolute_error,
            "resource_caps_present": state.resource_caps is not None,
            "numerical_error_bound_present": state.max_absolute_error is not None,
        },
        "reward": None,
        "native_metrics": None,
        "no_raw_model_weights": True,
        "no_hls_or_fpga_feasibility_claim": True,
    }


class HardwarePlanningEnvironment:
    """RL-front-end planning session; NOT a synthetic-reward Gym environment.

    An actual training Env must bind these actions to original native HLS
    receipts and an independently audited latency-only reward first.
    """
    def __init__(self, model: FunctionalModel, *, part: str,
                 limits: TargetLimits = TargetLimits(max_parallel_units=8),
                 resource_caps: tuple[int, int, int, int, int] | None = None,
                 max_absolute_error: float | None = None):
        self.model = model
        self.part = part
        self.limits = limits
        self.resource_caps = resource_caps
        self.max_absolute_error = max_absolute_error
        self.current: HardwareState | None = None
        self.source_steps = 0

    def reset(self, *, requested_clock_ns: float = 10.0) -> tuple[dict, dict]:
        self.current = initial_state(
            self.model, part=self.part, clock_ns=requested_clock_ns,
            limits=self.limits, resource_caps=self.resource_caps,
            max_absolute_error=self.max_absolute_error,
        )
        self.source_steps = 0
        return graph_observation(self.current), {
            "state_sha256": self.current.state_sha256,
            "source_steps": 0,
            "native_vitis_calls": 0,
            "hardware_reward": None,
        }

    def offers(self, *, max_actions: int = 512) -> tuple[tuple[HardwareEdit, ...], dict]:
        if self.current is None:
            raise RuntimeError("Call reset() before querying an RL action menu")
        actions = candidate_actions(self.current, max_actions=max_actions)
        return actions, {
            "schema": SCHEMA,
            "source_only_action_mask": action_mask(self.current, actions),
            "action_features": action_features(self.current, actions),
            "continuous_clock_domain": knob_inventory(
                self.current
            )["global_continuous_clock"],
            "reward": None,
            "native_vitis_calls": 0,
        }

    def propose(self, action: HardwareEdit) -> tuple[dict, dict]:
        if self.current is None:
            raise RuntimeError("Call reset() before proposing source hardware edits")
        decision = transition(self.current, action)
        if decision.status == "SOURCE_ACCEPTED":
            self.current = decision.next
            self.source_steps += 1
        return graph_observation(self.current), decision.receipt()

    def step(self, action):
        raise RuntimeError(
            "No fake reward: only proposal planning is implemented. "
            "Bind audited original Vitis HLS evaluations before Gym step()/training."
        )


def main(argv=None) -> int:
    """Read-only ONNX design-space discovery; no source build folder or vendor calls."""
    import argparse
    from fpgai_research.frontend.onnx import load_onnx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--part", required=True, help="Explicit exact target FPGA silicon part")
    parser.add_argument("--clock-ns", type=float, default=10.0)
    parser.add_argument("--max-parallel-units", type=int, default=8)
    parser.add_argument("--resource-caps", type=int, nargs=5, metavar="N",
                        help="Explicit LUT FF DSP BRAM18K URAM budgets; omit if unknown")
    parser.add_argument("--max-absolute-error", type=float,
                        help="Explicit maximum original reference absolute error")
    args = parser.parse_args(argv)
    try:
        model = load_onnx(args.onnx)
        state = initial_state(
            model, part=args.part, clock_ns=args.clock_ns,
            limits=TargetLimits(max_parallel_units=args.max_parallel_units),
            resource_caps=tuple(args.resource_caps) if args.resource_caps else None,
            max_absolute_error=args.max_absolute_error,
        )
        result = {
            "schema": SCHEMA,
            "state_sha256": state.state_sha256,
            "source_receipt": validate_source(state),
            "knobs": knob_inventory(state),
            "observation": graph_observation(state),
            "native_experiments_started": 0,
            "training_started": False,
        }
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"RL hardware planning blocked: {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
