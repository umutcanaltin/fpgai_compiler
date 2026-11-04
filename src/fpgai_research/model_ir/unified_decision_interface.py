"""R2p: common policy-neutral C3 facade for two *bounded* CST compiler domains.

This integrates the previously source-backed sequential row-chain and
2..4-reader MatMul fanout decision adapters WITHOUT reproducing their
architecture resolver, source certificate, C++ code generator or GCC
verification. It does not implement arbitrary DAG legality, physical
resource masks, reward functions, policy training or model transfer.

This is a *facade* over two source compilers, NOT a proof that their
semantics, observation feature distributions or source memory objectives
are identical. Source admission, bounded E0 and actual GCC E1 remain
distinct. Unsupported states and ill-typed actions fail closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Any

from fpgai_research.model_ir.decision_interface import (
    Accepted as SequentialAccepted,
    DecisionAction,
    DecisionState,
    Rejected as SequentialRejected,
    graph_observation as sequential_observation,
    source_executed_action_mask as sequential_source_mask,
    state_hash as sequential_hash,
    transition as sequential_transition,
)
from fpgai_research.model_ir.fanout_decision_interface import (
    FanoutAccepted,
    FanoutDecisionState,
    FanoutRejected,
    fanout_state_hash,
    fanout_transition,
    multi_reader_graph_observation,
    source_qualified_multi_reader_mask,
)

SCHEMA = "fpgai-research/cst-R2p-unified-bounded-C3-interface/v1"
DOMAINS = ("sequential", "multi_reader_fanout")
MAX_ACTIONS = 8


def _sha64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _domain(state: DecisionState | FanoutDecisionState) -> str:
    if isinstance(state, DecisionState):
        return DOMAINS[0]
    if isinstance(state, FanoutDecisionState):
        return DOMAINS[1]
    raise TypeError("R2p requires an existing typed bounded compiler state")


def unified_state_hash(state: DecisionState | FanoutDecisionState) -> str:
    """Preserve the exact old compiler source state hash, without a new oracle."""
    return sequential_hash(state) if _domain(state) == DOMAINS[0] else fanout_state_hash(state)


def _bounded_actions(actions: tuple[DecisionAction, ...]) -> None:
    if (type(actions) is not tuple or len(actions) > MAX_ACTIONS
            or any(not isinstance(a, DecisionAction) for a in actions)):
        raise TypeError("R2p requires <=8 typed action candidates in a tuple")


def _verified_sequential_e0(cert: dict, state: DecisionState) -> bool:
    mapped = cert.get("mapped_value_contract")
    logical = cert.get("mapped_logical_event_contract")
    return bool(
        state.exchange_policy == "row_chain"
        and isinstance(mapped, dict)
        and mapped.get("schema") == "fpgai-research/cst-mapped-value-ir/v1"
        and type(mapped.get("value_elements")) is int
        and mapped["value_elements"] > 0
        and isinstance(logical, dict)
        and logical.get("status") == "VERIFIED"
    )


@dataclass(frozen=True)
class UnifiedDecision:
    domain: str
    status: str  # REJECTED, UNKNOWN_E0, or BOUNDED_CERTIFIED_E0
    action: DecisionAction
    original_state: DecisionState | FanoutDecisionState
    next_state: DecisionState | FanoutDecisionState
    parent_state_sha256: str
    source_admissible: bool
    e0_certified: bool
    state_sha256: str | None = None
    cpp_sha256: str | None = None
    certificate_sha256: str | None = None
    reason: str | None = None

    def as_receipt(self) -> dict:
        """Return only bounded, provenance-bearing policy-neutral summary."""
        return {
            "schema": SCHEMA, "domain": self.domain, "status": self.status,
            "action": asdict(self.action),
            "parent_state_sha256": self.parent_state_sha256,
            "source_admissible": self.source_admissible,
            "e0_certified": self.e0_certified,
            "state_sha256": self.state_sha256,
            "cpp_sha256": self.cpp_sha256,
            "certificate_sha256": self.certificate_sha256,
            "reason": self.reason,
            "native_hls_feasible": None,
            "physical_feasibility_checked": False,
        }


def unified_transition(
    state: DecisionState | FanoutDecisionState,
    action: DecisionAction,
) -> UnifiedDecision:
    """Run the one actual domain-specific source compiler transition once."""
    domain = _domain(state)
    if not isinstance(action, DecisionAction):
        raise TypeError("R2p requires an existing typed DecisionAction")
    before = unified_state_hash(state)
    if domain == DOMAINS[0]:
        result = sequential_transition(state, action)
        rejected = isinstance(result, SequentialRejected)
        accepted = isinstance(result, SequentialAccepted)
    else:
        result = fanout_transition(state, action)
        rejected = isinstance(result, FanoutRejected)
        accepted = isinstance(result, FanoutAccepted)
    if not (rejected or accepted):
        raise ValueError("R2p unexpected actual compiler decision type")
    if rejected:
        if result.state is not state:
            raise ValueError("R2p rejected action mutated original state")
        return UnifiedDecision(
            domain, "REJECTED", action, state, state, before,
            source_admissible=False, e0_certified=False, reason=result.reason,
        )
    cert = result.certificate
    after = unified_state_hash(result.state)
    if (cert.get("state_sha256") != after
            or cert.get("parent_state_sha256") != before
            or not _sha64(cert.get("cpp_sha256"))
            or not _sha64(cert.get("certificate_sha256"))
            or after == before):
        raise ValueError("R2p source lineage or unchanged-state certificate invalid")
    e0 = (
        _verified_sequential_e0(cert, result.state)
        if domain == DOMAINS[0]
        else cert.get("status") == "E0_SOURCE_EMITTED_AND_MULTI_READER_CERTIFIED"
    )
    return UnifiedDecision(
        domain, "BOUNDED_CERTIFIED_E0" if e0 else "UNKNOWN_E0",
        action, state, result.state, before,
        source_admissible=True, e0_certified=bool(e0),
        state_sha256=after, cpp_sha256=cert["cpp_sha256"],
        certificate_sha256=cert["certificate_sha256"],
        reason=None if e0 else "Actual compiler source accepted; E0 certificate absent",
    )


def unified_action_mask(
    state: DecisionState | FanoutDecisionState,
    candidates: tuple[DecisionAction, ...],
) -> dict:
    """Finite typed mask, one actual transition per action; zero GCC calls."""
    domain = _domain(state)
    _bounded_actions(candidates)
    parent = unified_state_hash(state)
    decisions = tuple(unified_transition(state, action) for action in candidates)
    if unified_state_hash(state) != parent:
        raise ValueError("R2p mask changed parent source state")
    return {
        "schema": SCHEMA, "domain": domain,
        "parent_state_sha256": parent, "candidate_count": len(candidates),
        "statuses": tuple(item.status for item in decisions),
        "source_admissible": tuple(item.source_admissible for item in decisions),
        "e0_certified": tuple(item.e0_certified for item in decisions),
        "rows": [item.as_receipt() for item in decisions],
        "original_gcc_calls": 0,
        "native_vitis_calls": 0,
        "original_vivado_calls": 0,
        "rl_policies_trained": 0,
        "physical_feasibility_checked": False,
        "scope": "sequential row-chain or exact static 2..4-reader fanout",
    }


def unified_graph_observation(state: DecisionState | FanoutDecisionState) -> dict:
    """Normalize both REAL compiler graph observations to common typed fields.

    This intentionally removes original source SHA and source/weight payloads
    from the returned policy feature object. Original E0 source lineage is
    separately exposed by unified_state_hash/unified_transition receipts.
    Static graph features are NOT general permutation-equivariance, hardware
    time schedules, RL sample efficiency or C3 scientific priority.
    """
    domain = _domain(state)
    native = (
        sequential_observation(state)
        if domain == DOMAINS[0] else multi_reader_graph_observation(state)
    )
    if not isinstance(native, dict) or not isinstance(native.get("nodes"), list):
        raise ValueError("R2p original graph observation missing")
    reader_counts: dict[str, int] = {}
    edges = native.get("edges")
    if not isinstance(edges, list):
        raise ValueError("R2p original graph edges missing")
    for edge in edges:
        if edge.get("kind") == "consumes":
            reader_counts[edge["src"]] = reader_counts.get(edge["src"], 0) + 1
    nodes: list[dict[str, Any]] = []
    for node in native["nodes"]:
        if node.get("type") == "tensor":
            shape = node.get("shape")
            if not isinstance(shape, list) or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("R2p malformed typed tensor shape")
            nodes.append({
                "id": node["id"], "type": "tensor", "rank": len(shape),
                "shape": list(shape), "constant": node["constant"],
                "reader_count": reader_counts.get(node["id"], 0),
                "dtype": node["dtype"],
            })
        elif node.get("type") == "operator":
            probe = node.get("phi_probe_projection", node.get("phi_probe"))
            if not isinstance(probe, dict):
                raise ValueError("R2p original spatial Phi probe missing")
            nodes.append({
                "id": node["id"], "type": "operator",
                "kind": node["kind"],
                "iteration_shape": list(node["iteration_shape"]),
                "mapping": dict(node["mapping"]),
                "phi_spatial_probe": dict(probe),
            })
        else:
            raise ValueError("R2p unknown graph node type")
    details = (
        {
            "exchange_policy": native["exchange_policy"],
            "row_block": native["row_block"],
            "mapped_e0_evidence_status": native["boundary_evidence_status"],
        }
        if domain == DOMAINS[0]
        else {
            "source_scratch_layout": native["scratch_layout"],
            "source_scratch_unit_slots": native["source_scratch_unit_slots"],
            "fixed_order_optimal_unit_slots": native["fixed_order_optimal_slots"],
            "fanout_reader_count": native["fanout_readers"],
        }
    )
    return {
        "schema": "fpgai-research/cst-R2p-unified-weight-private-graph/v1",
        "domain": domain,
        "observation_kind": "anonymous-static-graph-no-weight-values",
        "nodes": nodes, "edges": [dict(edge) for edge in edges],
        "target_limits": dict(native["target_limits"]),
        "source_domain": details,
        "limitations": [
            "No raw weights, post-HLS metrics or training reward in observations",
            "Phi probes are spatial only, not FPGA cycle/event timing",
            "Different supported source domains are identified, not equated",
        ],
    }


def unified_source_mask(
    state: DecisionState | FanoutDecisionState,
    candidates: tuple[DecisionAction, ...], *,
    runtime_input,
    max_e1_executions: int = MAX_ACTIONS,
) -> dict:
    """Normalize bounded actual GCC E1 screens without replacing their checkers.

    Maximum candidates <= max_e1_executions <= 8 is deliberate: R2n and
    R2o have distinct internal partial-budget semantics. The shared
    facade never silently drops a candidate or reports an unexecuted action
    as E1. Existing domain-specific APIs still support their old budgets.
    """
    domain = _domain(state)
    _bounded_actions(candidates)
    if (type(max_e1_executions) is not int
            or not 1 <= max_e1_executions <= MAX_ACTIONS
            or len(candidates) > max_e1_executions):
        raise ValueError("R2p shared E1 budget must cover every candidate and be 1..8")
    before = unified_state_hash(state)
    raw = (
        sequential_source_mask(
            state, candidates, runtime_input=runtime_input,
            max_e1_executions=max_e1_executions)
        if domain == DOMAINS[0] else
        source_qualified_multi_reader_mask(
            state, candidates, runtime_input=runtime_input,
            max_e1_executions=max_e1_executions)
    )
    if unified_state_hash(state) != before:
        raise ValueError("R2p source screen changed original compiler state")
    if (not isinstance(raw, dict)
            or raw.get("candidate_count") != len(candidates)
            or not isinstance(raw.get("rows"), list)
            or len(raw["rows"]) != len(candidates)
            or raw.get("rl_policies_trained") != 0):
        raise ValueError("R2p incomplete or invalid original source GCC outcomes")
    if domain == DOMAINS[0]:
        if (raw.get("native_vitis_calls") != 0
                or raw.get("physical_feasibility_checked") is not False):
            raise ValueError("R2p sequential source evidence claimed native hardware")
    else:
        if (raw.get("original_vitis_calls") != 0
                or raw.get("original_vivado_calls") != 0
                or raw.get("hardware_feasibility_checked") is not False):
            raise ValueError("R2p multi-reader source evidence claimed native hardware")
    status_map = {
        "E1_GCC_SOURCE_QUALIFIED": "E1_GCC_QUALIFIED",
        "E1_GCC_MULTI_READER_QUALIFIED": "E1_GCC_QUALIFIED",
        "REJECTED": "REJECTED",
        "UNKNOWN_E0_CERTIFICATE": "UNKNOWN_E0",
        "UNKNOWN_E1_NOT_RUN": "UNKNOWN_E1_NOT_RUN",
        "FAILED_E1_SOURCE_EXECUTION": "FAILED_E1",
    }
    rows = []
    for i, case in enumerate(raw["rows"]):
        native_status = case.get("status")
        if native_status not in status_map:
            raise ValueError("R2p unexpected original GCC action status")
        admissible = case.get("source_admissible")
        e0 = (case.get("e0_mapped_certified") if domain == DOMAINS[0]
              else case.get("e0_certified"))
        e1 = (case.get("e1_host_source_qualified") if domain == DOMAINS[0]
              else case.get("e1_qualified"))
        if (type(admissible) is not bool or type(e0) is not bool
                or type(e1) is not bool):
            raise ValueError("R2p original source action evidence type invalid")
        normalized = status_map[native_status]
        if (e1 != (normalized == "E1_GCC_QUALIFIED")
                or (e1 and not (admissible and e0))
                or (not admissible and e0)
                or (normalized == "REJECTED" and (admissible or e0))
                or (normalized == "UNKNOWN_E0" and
                    (not admissible or e0))
                or (normalized in ("UNKNOWN_E1_NOT_RUN","FAILED_E1")
                    and (not admissible or not e0))):
            raise ValueError("R2p cannot promote unknown/failed GCC evidence")
        if admissible and (
                not _sha64(case.get("state_sha256"))
                or not _sha64(case.get("cpp_sha256"))):
            raise ValueError("R2p accepted source action lacks SHA identity")
        if e1:
            if domain == DOMAINS[0]:
                if not all(_sha64(case.get(key)) for key in (
                        "qualification_sha256", "source_trace_sha256",
                        "host_input_sha256")):
                    raise ValueError("R2p original sequential GCC hashes absent")
                if any(type(case.get(key)) not in (int,float)
                       or not isfinite(case[key]) or case[key] < 0
                       for key in ("host_max_absolute_error","host_rmse")):
                    raise ValueError("R2p original sequential GCC numerical error malformed")
            else:
                receipt = case.get("gcc_receipt")
                if (not isinstance(receipt,dict)
                        or not all(_sha64(receipt.get(key)) for key in (
                            "source_cpp_sha256", "original_stdout_sha256",
                            "original_input_sha256",
                            "original_runtime_source_event_sha256"))
                        or receipt["source_cpp_sha256"] != case["cpp_sha256"]):
                    raise ValueError("R2p original multi-reader GCC receipt malformed")
        if (case.get("action_index", case.get("index")) != i
                or case.get("action") != asdict(candidates[i])):
            raise ValueError("R2p action identity/order changed during source qualification")
        rows.append({
            "action_index": i, "action": asdict(candidates[i]),
            "status": status_map[native_status],
            "original_status": native_status,
            "source_admissible": admissible,
            "e0_certified": e0,
            "e1_gcc_qualified": e1,
            "state_sha256": case.get("state_sha256"),
            "cpp_sha256": case.get("cpp_sha256"),
            "reason": case.get("reason"),
            "gcc_receipt": (
                {"source_qualification_sha256": case.get("qualification_sha256"),
                 "source_trace_sha256": case.get("source_trace_sha256"),
                 "original_input_sha256": case.get("host_input_sha256"),
                 "host_max_absolute_error": case.get("host_max_absolute_error"),
                 "host_rmse": case.get("host_rmse")}
                if domain == DOMAINS[0] and e1
                else case.get("gcc_receipt") if e1 else None
            ),
        })
    return {
        "schema": "fpgai-research/cst-R2p-unified-GCC-E1-mask/v1",
        "domain": domain, "parent_state_sha256": before,
        "candidate_count": len(candidates),
        "statuses": tuple(r["status"] for r in rows),
        "e1_qualified": tuple(r["e1_gcc_qualified"] for r in rows),
        "e0_certified": tuple(r["e0_certified"] for r in rows),
        "rows": rows,
        "gcc_e1_attempts": (
            raw["e1_executions_attempted"]
            if domain == DOMAINS[0] else raw["gcc_e1_attempts"]
        ),
        "native_vitis_calls": 0, "original_vivado_calls": 0,
        "rl_policies_trained": 0, "physical_feasibility_checked": False,
        "scope": "actual GCC host-float single input, not native/FPGA/learned RL",
    }
