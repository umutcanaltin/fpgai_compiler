"""Policy-independent, certificate-backed CST search interface (E0/E1 subset).

The same architecture resolver and source compiler decide EVERY action,
regardless of whether it came from a human, random search, or a future RL
policy. A mask means that this exact *complete-design* action compiled with
the existing bounded verification gate. It does NOT prove original Vitis
resources, HLS clock timing, or general Phi-to-element readiness.

No Gym, Torch, native HLS, clocks-as-rewards or RL files are imported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import log1p
from typing import Any, Mapping

from fpgai_research.architecture.resolver import TargetLimits, resolve, coordinate
from fpgai_research.compiler.cst_pipeline import compile_cst
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, ArchitecturalModel, FunctionalModel,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)

SCHEMA = "fpgai-research/cst-certified-decision-interface/v1"
EDITABLE = (
    "pes", "simd", "parallelism", "serialization",
    "pipeline_ii", "tile", "precision_bits", "integer_bits",
)
EXCHANGE_POLICIES = ("whole_tensor", "row_chain", "row_lifetime", "column_major")


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DecisionState:
    """Full, valid static compiler candidate; no partially assigned knobs."""
    model: FunctionalModel
    requests: tuple[tuple[str, ArchitectureRequest], ...]
    exchange_policy: str = "whole_tensor"
    row_block: int = 1
    intermediate_tensor: str | None = None
    limits: TargetLimits = TargetLimits()

    def __post_init__(self):
        if self.exchange_policy not in EXCHANGE_POLICIES:
            raise ValueError("Unsupported decision-state exchange policy")
        if len(self.requests) != len(self.model.operators):
            raise ValueError("Every operator requires a complete architecture request")
        expected = tuple(op.name for op in self.model.operators)
        if tuple(name for name, _ in self.requests) != expected:
            raise ValueError("Operator requests must match topological identities")
        if any(not isinstance(req, ArchitectureRequest) for _,req in self.requests):
            raise TypeError("Malformed architecture request")
        if type(self.row_block) is not int or self.row_block < 1:
            raise ValueError("row_block must be a positive integer")
        if not isinstance(self.limits, TargetLimits):
            raise TypeError("Expected device limits from compiler resolver")


@dataclass(frozen=True)
class DecisionAction:
    scope: str
    target: str
    field: str
    value: object


@dataclass(frozen=True)
class Accepted:
    state: DecisionState
    certificate: dict
    status: str = "ACCEPTED"


@dataclass(frozen=True)
class Rejected:
    state: DecisionState
    reason: str
    code: str
    counterexample: dict
    status: str = "REJECTED"


def state_hash(state: DecisionState) -> str:
    """Exact source-state hash, includes all weights through Functional SHA."""
    return _digest({
        "schema": SCHEMA,
        "functional_sha256": functional_fingerprint(state.model),
        "requests": [(name,asdict(req)) for name, req in state.requests],
        "policy":state.exchange_policy, "row_block":state.row_block,
        "intermediate_tensor":state.intermediate_tensor,
        "target_limits":asdict(state.limits),
    })


def validate_state(state: DecisionState):
    """Return the ACTUAL C++ emission after compiler checks, not a proxy."""
    if not isinstance(state, DecisionState):
        raise TypeError("Expected typed DecisionState")
    default = ArchitectureRequest()
    arch = resolve(state.model, default, dict(state.requests), state.limits)
    compiled = compile_cst(
        arch, exchange_policy=state.exchange_policy,
        intermediate_tensor=state.intermediate_tensor,
        row_block=state.row_block)
    if compiled.emission.manifest["functional_sha256"] != functional_fingerprint(state.model):
        raise AssertionError("Compiler changed Functional graph")
    return arch, compiled


def certificate(state: DecisionState, *, parent_hash: str | None = None) -> dict:
    arch, compiled = validate_state(state)
    schedule = compiled.evidence.get("schedule_witness")
    if schedule is not None and schedule.get("status") != "VERIFIED":
        raise ValueError("Exact row schedule lacks accepted finite certificate")
    mapped = None
    logical_events = None
    if state.exchange_policy == "row_chain":
        from fpgai_research.model_ir.mapped_value_contract import (
            mapped_value_receipt, resolve_mapped_values)
        try:
            mapped = mapped_value_receipt(resolve_mapped_values(
                arch, row_block=state.row_block))
            from fpgai_research.model_ir.mapped_event_schedule import (
                canonical_value_events, verify_value_events)
            proposed = canonical_value_events(arch, row_block=state.row_block)
            logical_events = verify_value_events(
                arch, proposed, row_block=state.row_block)
            if logical_events["status"] != "VERIFIED":
                raise ValueError("E0 logical value-event schedule rejected: " +
                                 str(logical_events.get("reason")))
        except ValueError as exc:
            # E0 proof bounds may be exceeded for an otherwise source-valid
            # large model. Such a mask remains source-admission only;
            # mapped-value qualification is UNKNOWN, never a false proof.
            if "bound" not in str(exc):
                raise
            mapped = {"status":"UNVERIFIED_BOUND",
                      "reason":str(exc), "evidence":"E0-unverified"}
    result = {
        "schema": SCHEMA,
        "status": "SOURCE_EMITTED_AND_COMPILER_CONTRACT_ACCEPTED",
        "state_sha256": state_hash(state),
        "parent_state_sha256": parent_hash,
        "functional_sha256":functional_fingerprint(state.model),
        "effective_architecture_sha256":architecture_fingerprint(arch),
        "cpp_sha256":compiled.emission.manifest["cpp_sha256"],
        "exchange_policy":state.exchange_policy,
        "row_block":state.row_block,
        "schedule_verification_status":schedule.get("status") if schedule else "NOT_APPLICABLE",
        "schedule_sha256":schedule.get("schedule_sha256") if schedule else None,
        "mapped_value_contract": mapped,
        "mapped_logical_event_contract": logical_events,
        "source_scratch_elements":compiled.emission.manifest.get(
            "source_intermediate_scratch_elements"),
        "native_vitis_calls":0,
        "evidence":"E0-compiler-validation-plus-E1-source-generated-not-host-executed",
        "limitations":[
            "Whole-row stage atomicity remains the row-chain schedule model",
            "Phi-to-per-element readiness is not independently proved here",
            "Source C++ emission is not an original HLS physical feasibility claim",
            "No RL policy or reward function is present",
        ],
    }
    result["certificate_sha256"] = _digest(result)
    return result


def create_state(model: FunctionalModel, *,
                 default: ArchitectureRequest = ArchitectureRequest(),
                 overrides: Mapping[str, ArchitectureRequest] | None = None,
                 limits: TargetLimits = TargetLimits(),
                 exchange_policy: str = "whole_tensor",
                 intermediate_tensor: str | None = None,
                 row_block: int = 1) -> DecisionState:
    """Require that the initial complete candidate is actually source-valid."""
    model.validate()
    overrides = dict(overrides or {})
    if set(overrides) - {op.name for op in model.operators}:
        raise ValueError("Unknown operator-specific architecture override")
    state = DecisionState(
        model=model, requests=tuple(
            (op.name,overrides.get(op.name,default)) for op in model.operators),
        exchange_policy=exchange_policy, row_block=row_block,
        intermediate_tensor=intermediate_tensor, limits=limits)
    validate_state(state)
    return state


def _patched(state: DecisionState, action: DecisionAction) -> DecisionState:
    if not isinstance(action, DecisionAction):
        raise TypeError("Expected typed DecisionAction")
    if action.scope == "operator":
        if action.target not in {name for name,_ in state.requests}:
            raise ValueError("UNKNOWN_OPERATOR: No such operator identity")
        if action.field not in EDITABLE:
            raise ValueError("UNSUPPORTED_FIELD: Not an editable operator mapping")
        requests = []
        for name, request in state.requests:
            if name != action.target:
                requests.append((name,request))
                continue
            val = action.value
            if action.field == "tile":
                if not isinstance(val, (tuple,list)) or any(type(x) is not int for x in val):
                    raise ValueError("INVALID_VALUE: tile must contain integer extents")
                val = tuple(val)
            if action.field == "pipeline_ii":
                if val is not None and type(val) is not int:
                    raise ValueError("INVALID_VALUE: pipeline_ii expects integer or null")
            elif action.field != "tile" and type(val) is not int:
                raise ValueError("INVALID_VALUE: integer architecture choice required")
            requests.append((name,replace(request,**{action.field:val})))
        return replace(state,requests=tuple(requests))
    if action.scope == "exchange":
        if action.target not in ("", "model"):
            raise ValueError("UNSUPPORTED_TARGET: Exchange plan is global for this prototype")
        if action.field == "row_block":
            if type(action.value) is not int:
                raise ValueError("INVALID_VALUE: row_block must be integer")
            return replace(state,row_block=action.value)
        if action.field == "exchange_policy":
            if type(action.value) is not str or action.value not in EXCHANGE_POLICIES:
                raise ValueError("INVALID_POLICY: Unknown source exchange policy")
            return replace(state,exchange_policy=action.value)
        if action.field == "intermediate_tensor":
            if action.value is not None and (
                    type(action.value) is not str or not action.value):
                raise ValueError("INVALID_VALUE: tensor must be a nonempty identity or null")
            return replace(state,intermediate_tensor=action.value)
        raise ValueError("UNSUPPORTED_FIELD: Exchange field cannot be changed")
    raise ValueError("UNSUPPORTED_SCOPE: Allowed scopes are operator and exchange")


def transition(state: DecisionState, action: DecisionAction) -> Accepted | Rejected:
    """Never repair a candidate. On rejection, return the original state."""
    if not isinstance(state, DecisionState):
        raise TypeError("Expected typed DecisionState")
    try:
        next_state = _patched(state, action)
        if state_hash(next_state) == state_hash(state):
            raise ValueError("NO_CHANGE: Proposed edit has no effect")
        receipt = certificate(next_state, parent_hash=state_hash(state))
    except (ValueError, TypeError, OverflowError, IndexError) as exc:
        code = getattr(exc,"code",None) or (
            str(exc).split(":",1)[0] if ":" in str(exc) else type(exc).__name__)
        return Rejected(state=state,reason=str(exc),code=str(code),
                        counterexample={
                            "scope": getattr(action,"scope",None),
                            "target": getattr(action,"target",None),
                            "field": getattr(action,"field",None),
                            "requested_value":repr(getattr(action,"value",None)),
                            "reason":str(exc),
                        })
    return Accepted(state=next_state,certificate=receipt)


def action_mask(state: DecisionState,
                candidates: tuple[DecisionAction, ...]) -> dict:
    """A mask is exact transition success for these FINITE candidate actions.

    The legal set has no guarantee of HLS resource/timing feasibility and
    the catalog is not claimed to exhaust all possible architectures.
    """
    if type(candidates) is not tuple or any(
            not isinstance(a,DecisionAction) for a in candidates):
        raise TypeError("Provide a finite tuple of typed actions")
    results = tuple(transition(state,action) for action in candidates)
    return {
        "schema":SCHEMA,
        "state_sha256":state_hash(state),
        "legal":tuple(r.status == "ACCEPTED" for r in results),
        "codes":tuple("ACCEPTED" if r.status == "ACCEPTED" else r.code for r in results),
        "candidate_count":len(candidates),
        "verified_source_semantics_only":True,
        "native_hls_feasibility_checked":False,
    }



def certified_action_mask(state: DecisionState,
                          candidates: tuple[DecisionAction, ...]) -> dict:
    """Three-valued, policy-independent bounded mapped-value action gate.

    action_mask remains a SOURCE-admission mask for compatibility. This
    stronger mask labels ACCEPTED candidates UNKNOWN unless the actual
    mapped-value E0 relation is certified; UNKNOWN is never legal. It does
    NOT prove partial-reduction timing, hardware resources or fan-out.
    """
    if type(candidates) is not tuple or any(
            not isinstance(action, DecisionAction) for action in candidates):
        raise TypeError("Provide a finite tuple of typed actions")
    # One actual compiler transition per proposal: mask and certificate
    # statuses MUST refer to the identical qualification attempt.
    decisions = tuple(transition(state, action) for action in candidates)
    statuses = []
    reasons = []
    for result in decisions:
        if isinstance(result, Rejected):
            statuses.append("REJECTED")
            reasons.append(result.reason)
        else:
            mapped = result.certificate.get("mapped_value_contract")
            if (result.state.exchange_policy == "row_chain"
                    and isinstance(mapped, dict)
                    and mapped.get("schema") == "fpgai-research/cst-mapped-value-ir/v1"
                    and mapped.get("value_elements", 0) > 0
                    and result.certificate.get(
                        "mapped_logical_event_contract", {}).get("status") == "VERIFIED"):
                statuses.append("BOUNDED_MAPPED_VALUE")
                reasons.append(None)
            else:
                statuses.append("UNKNOWN")
                reasons.append("No qualified mapped tensor-value certificate for this action")
    return {
        "schema": "fpgai-research/cst-mapped-decision-mask/v1",
        "state_sha256": state_hash(state),
        "candidate_count": len(candidates),
        "status": tuple(statuses),
        "legal": tuple(s == "BOUNDED_MAPPED_VALUE" for s in statuses),
        "source_admissible": tuple(isinstance(r, Accepted) for r in decisions),
        "reasons": tuple(reasons),
        "native_hls_feasibility_checked": False,
        "scope": "bounded spatial value ownership; not element-time or hardware proof",
    }



def source_executed_action_mask(
    state: DecisionState,
    candidates: tuple[DecisionAction, ...],
    *,
    runtime_input,
    max_e1_executions: int = 8,
) -> dict:
    """E1-qualified *finite* action mask using the ACTUAL GCC source path.

    One production compiler transition per candidate, followed by E0
    mapped-value evidence and actual GCC float-numerical + value-lifetime
    execution for E0-qualified row-chain actions only. This strengthens
    source evidence relative to the E0-only certified_action_mask, but
    is NOT a proof of arbitrary inputs, physical resource feasibility,
    fixed-point equivalence, HLS timing, bank ports, or RL performance.

    Failed/unknown cases stay visible in their original action order.
    In particular SOURCE-admissible WHOLE_TENSOR is not promoted to
    E1-certified row-chain value-lifetime semantics.
    """
    if not isinstance(state, DecisionState):
        raise TypeError("Expected typed complete design state")
    if type(candidates) is not tuple or any(
            not isinstance(action, DecisionAction) for action in candidates):
        raise TypeError("Provide a finite tuple of typed actions")
    if type(max_e1_executions) is not int or not 1 <= max_e1_executions <= 64:
        raise ValueError("Positive bounded E1 qualification budget required")
    if runtime_input is None:
        raise ValueError("Explicit fixed host input required for source E1 qualification")
    # Fail before any GCC execution if the upper bound is impossible.
    if len(candidates) > max_e1_executions:
        raise ValueError("Candidate count exceeds original GCC qualification budget")
    import numpy as np
    import subprocess
    input_array = np.asarray(runtime_input, dtype=np.float32)
    expected_shape = state.model.tensor(state.model.inputs[0]).shape
    if input_array.shape != expected_shape or not np.all(np.isfinite(input_array)):
        raise ValueError("Finite runtime input must match original Functional shape")
    result_rows = []
    e1_attempted = 0
    for index, action in enumerate(candidates):
        result = transition(state, action)
        row = {
            "action_index": index,
            "action": asdict(action),
            "source_admissible": isinstance(result, Accepted),
            "e0_mapped_certified": False,
            "e1_host_source_qualified": False,
            "status": "UNKNOWN",
            "reason": None,
            "state_sha256": None,
            "cpp_sha256": None,
            "qualification_sha256": None,
            "host_input_sha256": None,
            "host_max_absolute_error": None,
            "host_rmse": None,
            "source_trace_sha256": None,
            "last_reader_releases_verified": None,
        }
        if isinstance(result, Rejected):
            row["status"] = "REJECTED"
            row["reason"] = result.reason
        else:
            certificate_row = result.certificate
            row["state_sha256"] = certificate_row["state_sha256"]
            row["cpp_sha256"] = certificate_row["cpp_sha256"]
            mapped = certificate_row.get("mapped_value_contract")
            logical = certificate_row.get("mapped_logical_event_contract")
            row["e0_mapped_certified"] = bool(
                result.state.exchange_policy == "row_chain"
                and isinstance(mapped, dict)
                and mapped.get("schema") == "fpgai-research/cst-mapped-value-ir/v1"
                and type(mapped.get("value_elements")) is int
                and mapped["value_elements"] > 0
                and isinstance(logical, dict)
                and logical.get("status") == "VERIFIED")
            if not row["e0_mapped_certified"]:
                row["status"] = "UNKNOWN_E0_CERTIFICATE"
                row["reason"] = (
                    "Source admission is not a bounded mapped-value proof")
            else:
                e1_attempted += 1
                try:
                    evidence = qualify_accepted_source(
                        result, runtime_input=input_array, verify_value_trace=True)
                except (ValueError, RuntimeError, AssertionError,
                        OSError, TimeoutError, subprocess.TimeoutExpired) as err:
                    row["status"] = "FAILED_E1_SOURCE_EXECUTION"
                    row["reason"] = (
                        f"{type(err).__name__}: {str(err)[:300]}")
                else:
                    host = evidence.get("host_numerical") or {}
                    trace = evidence.get("source_value_lifetime") or {}
                    # Provenance inconsistency is a fatal audit failure,
                    # not an expected missing/failed native observation.
                    if (evidence.get("state_sha256") != certificate_row["state_sha256"]
                            or evidence.get("cpp_sha256") != certificate_row["cpp_sha256"]
                            or host.get("cpp_sha256") != certificate_row["cpp_sha256"]
                            or trace.get("cpp_sha256") != certificate_row["cpp_sha256"]):
                        raise ValueError(
                            "Actual GCC source identity disagrees with compiler decision")
                    if (host.get("accepted") is True
                            and trace.get("status") == "SOURCE_VALUE_LIFETIMES_VERIFIED"
                            and evidence.get("strongest_evidence")
                                == "E1-executed-GCC-value-and-numerical"
                            and type(trace.get("last_reader_releases_verified")) is int
                            and trace["last_reader_releases_verified"]
                                == mapped["value_elements"]):
                        row["status"] = "E1_GCC_SOURCE_QUALIFIED"
                        row["e1_host_source_qualified"] = True
                        row["qualification_sha256"] = evidence["qualification_sha256"]
                        row["host_input_sha256"] = host["input_sha256"]
                        row["host_max_absolute_error"] = host["max_absolute_error"]
                        row["host_rmse"] = host["rmse"]
                        row["source_trace_sha256"] = trace["source_trace_sha256"]
                        row["last_reader_releases_verified"] = trace[
                            "last_reader_releases_verified"]
                    else:
                        row["status"] = "FAILED_E1_SOURCE_EXECUTION"
                        row["reason"] = "Actual GCC evidence was incomplete or unqualified"
        result_rows.append(row)
    return {
        "schema": "fpgai-research/cst-e1-source-qualified-action-mask/v1",
        "parent_state_sha256": state_hash(state),
        "candidate_count": len(candidates),
        "source_admissible": tuple(row["source_admissible"] for row in result_rows),
        "e0_certified": tuple(row["e0_mapped_certified"] for row in result_rows),
        "e1_qualified": tuple(row["e1_host_source_qualified"] for row in result_rows),
        "statuses": tuple(row["status"] for row in result_rows),
        "e1_executions_attempted": e1_attempted,
        "rl_policies_trained": 0,
        "native_vitis_calls": 0,
        "physical_feasibility_checked": False,
        "numeric_evidence": "executed GCC host-float on one explicit finite input",
        "scope": "bounded static single-reader row-chain source only",
        "rows": result_rows,
    }


def candidate_actions(state: DecisionState, *, max_actions: int = 128) -> tuple[DecisionAction,...]:
    """Deterministic bounded proposal catalog; not a completeness theorem."""
    if type(max_actions) is not int or not 1 <= max_actions <= 1024:
        raise ValueError("Valid finite action budget required")
    actions=[]
    for op in state.model.operators:
        for field,vals in (
            ("pes",(1,2,4)), ("simd",(1,2,4)),
            ("parallelism",(1,2)), ("serialization",(1,2)),
            ("pipeline_ii",(None,1,2)),
        ):
            for val in vals:
                actions.append(DecisionAction("operator",op.name,field,val))
    for block in sorted({1,2,state.model.tensor(state.model.inputs[0]).shape[0]}):
        actions.append(DecisionAction("exchange","model","row_block",block))
    for policy in ("whole_tensor","row_chain"):
        actions.append(DecisionAction("exchange","model","exchange_policy",policy))
    return tuple(actions[:max_actions])


def graph_observation(state: DecisionState) -> dict:
    """Stable typed, weight-free variable-sized graph for GNN feature adapters.

    Nodes get topologically derived IDs; raw operator/tensor names and trained
    constant values are not present in the public observation. Hash retains
    provenance internally but is NOT a numeric model feature.
    """
    arch,_ = validate_state(state)
    f=state.model
    tensor_order=list(f.inputs)
    for op in f.operators:
        for name in op.inputs:
            if name not in tensor_order:
                tensor_order.append(name)
        for name in op.outputs:
            if name not in tensor_order:
                tensor_order.append(name)
    for tensor in f.tensors:
        if tensor.name not in tensor_order:
            tensor_order.append(tensor.name)
    tid={name:"t"+str(i) for i,name in enumerate(tensor_order)}
    nodes=[]
    for i,name in enumerate(tensor_order):
        t=f.tensor(name)
        readers=sum(name in op.inputs for op in f.operators)
        nodes.append({
            "id":"t"+str(i),"type":"tensor",
            "rank":len(t.shape),"shape":list(t.shape),
            "log_shape":[round(log1p(d),8) for d in t.shape],
            "constant":t.data is not None,"reader_count":readers,
            "dtype":t.dtype,
        })
    edges=[]
    for i,(op,point) in enumerate(zip(f.operators,arch.points)):
        node_id="o"+str(i)
        snapshot=coordinate(point,tuple(0 for _ in point.iteration_shape))
        probe=coordinate(point,tuple(min(1, n-1) for n in point.iteration_shape))
        nodes.append({
            "id":node_id, "type":"operator","kind":op.kind,
            "iteration_shape":list(point.iteration_shape),
            "mapping":asdict(point.effective),
            "phi_zero_projection":{
                "tile":list(snapshot["tile"]), "replica":snapshot["replica"],
                "pe":snapshot["pe"],"lane":snapshot["lane"],
                "reuse_step":list(snapshot["reuse_step"]),
            },
            "phi_probe_projection":{
                "tile":list(probe["tile"]), "replica":probe["replica"],
                "pe":probe["pe"],"lane":probe["lane"],
                "reuse_step":list(probe["reuse_step"]),
            },
        })
        for position,name in enumerate(op.inputs):
            edges.append({"src":tid[name],"dst":node_id,"kind":"consumes",
                          "port":position})
        for name in op.outputs:
            edges.append({"src":node_id,"dst":tid[name],"kind":"produces"})
    # First-class mapping-conditioned boundary features for a variable-size
    # GNN. These are bounded spatial ownership/read relations, never HLS cycles
    # or proof of early per-element scheduling. Unavailable contracts remain
    # explicitly unverified, not zero-filled numerical features.
    boundary_status = "NOT_APPLICABLE"
    if state.exchange_policy == "row_chain":
        from fpgai_research.model_ir.mapped_value_contract import resolve_mapped_values
        try:
            mapped=resolve_mapped_values(arch,row_block=state.row_block)
            by_name={}
            for value in mapped.values:
                by_name.setdefault(value.tensor,[]).append(value)
            for name, values in by_name.items():
                tensor_node=next(n for n in nodes if n["id"]==tid[name])
                tensor_node["mapped_exchange"]={
                    "value_elements":len(values),
                    "producer_contributions":sum(
                        v.producer_required_contributions for v in values),
                    "consumer_read_sites":sum(
                        v.consumer_required_reads for v in values),
                    "physical_row_slots":state.row_block,
                    "producer_phi_projection_sha256":_digest([
                        v.producer_phi_owner_sha256 for v in values]),
                    "consumer_phi_projection_sha256":_digest([
                        v.consumer_phi_read_sha256 for v in values]),
                    "contract_evidence":"E0-spatial-value-projection-only",
                }
            boundary_status = "BOUNDED_PHI_SPATIAL_PROJECTIONS"
        except ValueError as exc:
            if "bound" not in str(exc):
                raise
            boundary_status = "UNVERIFIED_BOUND"
    return {
        "schema":SCHEMA,
        "observation_kind":"public-static-no-weights",
        "boundary_evidence_status":boundary_status,
        "nodes":nodes,"edges":edges,
        "exchange_policy":state.exchange_policy,"row_block":state.row_block,
        "target_limits":asdict(state.limits),
        "full_source_state_sha256":state_hash(state),
        "evidence":"E0-canonical-graph-projection",
        "warning":"Phi snapshot is a spatial mapping coordinate, NOT an event readiness/cycle proof",
    }


def qualify_accepted_source(accepted: Accepted, *, runtime_input=None,
                            verify_value_trace: bool = False) -> dict:
    """Optional ORIGINAL generated GCC C++ qualification of one accepted edit.

    Deliberately separate from action masking: no simulated reward, no Vitis,
    and no assumption that accepting a typed candidate implies numerically
    equivalent host execution or mapped per-element readiness evidence.
    """
    if not isinstance(accepted, Accepted):
        raise TypeError("Source qualification requires a previously Accepted edit")
    arch, compiled = validate_state(accepted.state)
    if compiled.emission.manifest["cpp_sha256"] != accepted.certificate["cpp_sha256"]:
        raise ValueError("Source changed between the accepted action and qualification")
    if state_hash(accepted.state) != accepted.certificate["state_sha256"]:
        raise ValueError("Decision state no longer matches accepted certificate")
    evidence = {
        "schema": "fpgai-research/cst-decision-source-qualification/v1",
        "state_sha256": state_hash(accepted.state),
        "cpp_sha256": compiled.emission.manifest["cpp_sha256"],
        "functional_sha256": functional_fingerprint(accepted.state.model),
        "source_value_lifetime": None,
        "host_numerical": None,
        "native_vitis_calls": 0,
    }
    if verify_value_trace:
        if accepted.state.exchange_policy != "row_chain":
            raise ValueError("Actual per-value trace currently requires sequential row_chain")
        from fpgai_research.validation.cpp_value_trace import trace_cpp_values
        observed = trace_cpp_values(compiled.ir, emission=compiled.emission)
        if observed["status"] != "SOURCE_VALUE_LIFETIMES_VERIFIED":
            raise ValueError("Original C++ source tensor value refinement not qualified")
        projection=accepted.certificate.get("mapped_value_contract")
        if projection and projection.get("value_elements") is not None:
            if observed["last_reader_releases_verified"] != projection["value_elements"]:
                raise ValueError("E0 mapping-conditioned lifetime differs from actual C++ source")
        evidence["source_value_lifetime"] = observed
    if runtime_input is not None:
        from fpgai_research.validation.host import validate_host
        measured = validate_host(accepted.state.model, compiled.emission,
                                 runtime_input, atol=1e-5, rtol=1e-4)
        if not measured["accepted"]:
            raise ValueError("Actual generated C++ does not match numerical model")
        evidence["host_numerical"] = measured
    evidence["strongest_evidence"] = (
        "E1-executed-GCC-value-and-numerical"
        if evidence["source_value_lifetime"] is not None and
           evidence["host_numerical"] is not None
        else "E1-executed-GCC-partial"
        if any(evidence[k] is not None for k in
               ("source_value_lifetime", "host_numerical"))
        else "E1-source-generated-only")
    evidence["qualification_sha256"] = _digest(evidence)
    return evidence
