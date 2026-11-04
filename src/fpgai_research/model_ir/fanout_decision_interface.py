"""R2o: policy-neutral source-backed decisions for bounded 2..4-reader fanout.

This extends C3 action semantics beyond the earlier single-reader row-chain
prototype. It reuses EXACT existing R2f E0 multi-reader Phi/last-use and
R2h actual address-alias C++ lowering and independent GCC/NumPy checker.
No native HLS, cycles, bank ports, hardware BRAM, trained RL or arbitrary
DAG semantics are implied. Complete (not partial) architecture states only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import log1p
from typing import Mapping

from fpgai_research.architecture.resolver import TargetLimits, resolve, coordinate
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    LAYOUTS, emit_allocated_fanout_cpp, source_layout,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel
from fpgai_research.model_ir.decision_interface import DecisionAction
from fpgai_research.model_ir.fanout_contract import admit_fanout
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)

SCHEMA = "fpgai-research/cst-R2o-multi-reader-C3-decisions/v1"


class SourceProvenanceError(ValueError):
    """R2o source SHA inconsistency: fatal, not an ordinary failed GCC case."""


def _sha(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FanoutDecisionState:
    model: FunctionalModel
    requests: tuple[tuple[str, ArchitectureRequest], ...]
    layout: str = "disjoint"
    limits: TargetLimits = TargetLimits()

    def __post_init__(self):
        if not isinstance(self.model, FunctionalModel):
            raise TypeError("R2o requires typed FunctionalModel")
        if self.layout not in LAYOUTS:
            raise ValueError("R2o requires disjoint/per_element_optimal scratch policy")
        if not isinstance(self.limits, TargetLimits):
            raise TypeError("R2o requires typed target limits")
        if (tuple(name for name, _ in self.requests)
                != tuple(op.name for op in self.model.operators)
                or any(not isinstance(req, ArchitectureRequest)
                       for _, req in self.requests)):
            raise ValueError("R2o complete operator requests must match Functional graph")


@dataclass(frozen=True)
class FanoutAccepted:
    state: FanoutDecisionState
    certificate: dict
    status: str = "ACCEPTED"


@dataclass(frozen=True)
class FanoutRejected:
    state: FanoutDecisionState
    reason: str
    code: str
    action: DecisionAction
    status: str = "REJECTED"


def fanout_state_hash(state: FanoutDecisionState) -> str:
    if not isinstance(state, FanoutDecisionState):
        raise TypeError("R2o requires typed fanout decision state")
    return _sha({
        "schema": SCHEMA,
        "functional_sha256": functional_fingerprint(state.model),
        "requests": [(name, asdict(req)) for name, req in state.requests],
        "layout": state.layout,
        "limits": asdict(state.limits),
    })


def _compile(state: FanoutDecisionState):
    arch = resolve(
        state.model, ArchitectureRequest(),
        dict(state.requests), state.limits)
    spec, assignments, capacity, optimal, events = source_layout(
        arch, layout=state.layout)
    emitted = emit_allocated_fanout_cpp(arch, layout=state.layout)
    manifest = emitted.manifest
    source_digest = sha256(emitted.cpp.encode()).hexdigest()
    if (manifest.get("source_cpp_sha256") != source_digest
            or manifest.get("functional_sha256")
                != functional_fingerprint(state.model)
            or manifest.get("architecture_sha256")
                != architecture_fingerprint(arch)
            or manifest.get("source_scratch_array_elements") != capacity
            or manifest.get("source_optimal_unit_slots_for_fixed_order")
                != optimal.unit_slots
            or manifest.get("source_event_order_sha256")
                != optimal.schedule_sha256
            or manifest.get("source_layout") != state.layout):
        raise ValueError("R2o original R2h source/E0 certificate lineage mismatch")
    if set(spec.readers) != {
            op.name for op in state.model.operators if op.name.startswith("reader")}:
        raise ValueError("R2o cannot silently change multi-reader topology")
    return arch, spec, emitted, capacity, optimal, assignments, events


def fanout_certificate(state: FanoutDecisionState, *,
                       parent_hash: str | None = None) -> dict:
    arch, spec, emission, slots, optimum, _assignments, events = _compile(state)
    result = {
        "schema": SCHEMA,
        "status": "E0_SOURCE_EMITTED_AND_MULTI_READER_CERTIFIED",
        "state_sha256": fanout_state_hash(state),
        "parent_state_sha256": parent_hash,
        "functional_sha256": functional_fingerprint(state.model),
        "architecture_sha256": architecture_fingerprint(arch),
        "cpp_sha256": emission.manifest["source_cpp_sha256"],
        "source_event_sha256": emission.manifest["source_event_order_sha256"],
        "source_slot_map_sha256": emission.manifest["source_slot_map_sha256"],
        "fanout_readers": len(spec.readers),
        "mapped_reader_names": list(spec.readers),
        "reader_pes": [p.effective.pes for p in arch.points[1:1+len(spec.readers)]],
        "layout": state.layout,
        "source_scratch_unit_slots": slots,
        "fixed_order_minimum_unit_slots": optimum.unit_slots,
        "source_event_count": len(events),
        "source_all_declared_mutable_array_elements":
            emission.manifest["source_all_declared_mutable_array_elements"],
        "evidence": "bounded-E0-multi-reader-Phi-and-source-slot-lifetime",
        "native_vitis_calls": 0,
        "new_vivado_calls": 0,
        "rl_policies_trained": 0,
        "physical_feasibility_checked": False,
    }
    result["certificate_sha256"] = _sha(result)
    return result


def create_fanout_state(model: FunctionalModel, *,
                        default: ArchitectureRequest = ArchitectureRequest(),
                        overrides: Mapping[str, ArchitectureRequest] | None = None,
                        layout: str = "disjoint",
                        limits: TargetLimits = TargetLimits()) -> FanoutDecisionState:
    model.validate()
    overrides = dict(overrides or {})
    if set(overrides) - {op.name for op in model.operators}:
        raise ValueError("R2o unknown operator request")
    state = FanoutDecisionState(
        model=model,
        requests=tuple((op.name, overrides.get(op.name, default))
                       for op in model.operators),
        layout=layout, limits=limits)
    fanout_certificate(state)
    return state


def _patched(state: FanoutDecisionState, action: DecisionAction) -> FanoutDecisionState:
    if not isinstance(action, DecisionAction):
        raise TypeError("R2o requires typed DecisionAction")
    if action.scope == "operator":
        readers = set(admit_fanout(resolve(
            state.model, ArchitectureRequest(), dict(state.requests),
            state.limits)).readers)
        if action.target not in readers:
            raise ValueError("UNSUPPORTED_OPERATOR: only independent fanout readers are editable")
        if action.field != "pes":
            raise ValueError("UNSUPPORTED_FIELD: R2o reader pilot edits only PE width")
        if type(action.value) is not int:
            raise ValueError("INVALID_VALUE: reader PE width must be integer")
        return replace(state, requests=tuple(
            (name, replace(req, pes=action.value)
             if name == action.target else req)
            for name, req in state.requests))
    if (action.scope == "memory" and action.target == "model"
            and action.field == "layout"):
        if type(action.value) is not str or action.value not in LAYOUTS:
            raise ValueError("INVALID_LAYOUT: choose disjoint or per_element_optimal")
        return replace(state, layout=action.value)
    raise ValueError("UNSUPPORTED_SCOPE: only reader PE and global scratch layout")


def fanout_transition(state: FanoutDecisionState,
                      action: DecisionAction) -> FanoutAccepted | FanoutRejected:
    if not isinstance(state, FanoutDecisionState):
        raise TypeError("R2o expects complete fanout state")
    if not isinstance(action, DecisionAction):
        raise TypeError("R2o expects typed architecture action")
    try:
        updated = _patched(state, action)
        if fanout_state_hash(updated) == fanout_state_hash(state):
            raise ValueError("NO_CHANGE: proposed edit does not change complete state")
        proof = fanout_certificate(updated, parent_hash=fanout_state_hash(state))
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        reason = str(exc)
        return FanoutRejected(
            state, reason, reason.split(":", 1)[0],
            action)
    return FanoutAccepted(updated, proof)


def multi_reader_graph_observation(state: FanoutDecisionState) -> dict:
    """Anonymous GNN-ready typed graph with reader Phi and E0 scratch metadata.

    The observation carries *no raw constant weight values or GCC/numeric
    outcomes*, and its spatial Phi samples are NOT hardware issue cycles.
    """
    arch, spec, _emission, capacity, optimal, _slots, _events = _compile(state)
    model = state.model
    tensors = [t.name for t in model.tensors]
    lookup = {name: "t"+str(i) for i, name in enumerate(tensors)}
    nodes = [
        {"id": lookup[t.name], "type": "tensor", "shape": list(t.shape),
         "dtype": t.dtype, "constant": t.data is not None,
         "log_shape": [round(log1p(x), 8) for x in t.shape]}
        for t in model.tensors
    ]
    edges = []
    for i, (op, point) in enumerate(zip(model.operators, arch.points)):
        pos = tuple(min(1, d-1) for d in point.iteration_shape)
        phi = coordinate(point, pos)
        nodes.append({
            "id": "o"+str(i), "type": "operator", "kind": op.kind,
            "iteration_shape": list(point.iteration_shape),
            "mapping": asdict(point.effective),
            "phi_probe": {
                "replica": phi["replica"], "pe": phi["pe"], "lane": phi["lane"],
                "reuse_step": list(phi["reuse_step"]),
            },
        })
        for port, name in enumerate(op.inputs):
            edges.append({"src": lookup[name], "dst": "o"+str(i),
                          "kind": "consumes", "port": port})
        for name in op.outputs:
            edges.append({"src": "o"+str(i), "dst": lookup[name],
                          "kind": "produces"})
    return {
        "schema": "fpgai-research/cst-R2o-multi-reader-C3-graph/v1",
        "observation_kind": "static-weight-private-E0-graph",
        "nodes": nodes, "edges": edges,
        "fanout_readers": len(spec.readers),
        "scratch_layout": state.layout,
        "source_scratch_unit_slots": capacity,
        "fixed_order_optimal_slots": optimal.unit_slots,
        "target_limits": asdict(state.limits),
        "full_source_state_sha256": fanout_state_hash(state),
        "evidence": "E0-spatial-Phi-and-multi-reader-source-value",
        "limitation": "No actual cycle/BRAM feasibility or trained RL",
    }


def source_qualified_multi_reader_mask(
    state: FanoutDecisionState,
    actions: tuple[DecisionAction, ...], *,
    runtime_input,
    max_e1_executions: int = 8,
    evidence_dir=None,
) -> dict:
    """One transition per policy-independent proposal; real R2h GCC checks.

    Every source-admissible R2o edit receives an E0 certificate by construction.
    UNKNOWN is retained for source-qualified but unexecuted proposals when
    the explicit GCC execution budget is exhausted. No original vendor calls.
    """
    if type(actions) is not tuple or any(
            not isinstance(a, DecisionAction) for a in actions):
        raise TypeError("R2o finite tuple of typed actions required")
    if type(max_e1_executions) is not int or not 0 <= max_e1_executions <= 32:
        raise ValueError("R2o bounded nonnegative GCC execution budget required")
    if len(actions) > 32:
        raise ValueError("R2o finite candidate action budget exceeded")
    import numpy as np
    import subprocess
    from pathlib import Path
    inp = np.asarray(runtime_input, dtype=np.float32)
    expected_shape = state.model.tensor(state.model.inputs[0]).shape
    if inp.shape != expected_shape or not np.all(np.isfinite(inp)):
        raise ValueError("R2o fixed input shape/finite values required")
    if evidence_dir is not None:
        folder = Path(evidence_dir)
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R2o original GCC trace evidence directory must exist")
    else:
        folder = None
    decisions = tuple(fanout_transition(state, action) for action in actions)
    rows = []
    attempted = 0
    for i, (action, d) in enumerate(zip(actions, decisions)):
        rec = {
            "index": i, "action": asdict(action),
            "source_admissible": isinstance(d, FanoutAccepted),
            "status": None, "e0_certified": False,
            "e1_qualified": False, "reason": None,
            "state_sha256": None, "cpp_sha256": None,
            "certificate_sha256": None,
            "gcc_receipt": None,
            "source_file": None, "source_manifest_file": None,
            "original_stdout_file": None,
        }
        if isinstance(d, FanoutRejected):
            rec["status"] = "REJECTED"
            rec["reason"] = d.reason
        else:
            cert = d.certificate
            rec.update({
                "e0_certified": True,
                "state_sha256": cert["state_sha256"],
                "cpp_sha256": cert["cpp_sha256"],
                "certificate_sha256": cert["certificate_sha256"],
            })
            if attempted >= max_e1_executions:
                rec["status"] = "UNKNOWN_E1_NOT_RUN"
                rec["reason"] = "Bounded original GCC execution budget exhausted"
            else:
                attempted += 1
                try:
                    arch = resolve(
                        d.state.model, ArchitectureRequest(),
                        dict(d.state.requests), d.state.limits)
                    emitted = emit_allocated_fanout_cpp(arch, layout=d.state.layout)
                    if (emitted.manifest["source_cpp_sha256"] != cert["cpp_sha256"]
                            or emitted.manifest["architecture_sha256"]
                                != cert["architecture_sha256"]):
                        raise SourceProvenanceError(
                            "Original source changed after E0 certificate")
                    if folder is not None:
                        source_path = folder / f"case_{i:03d}_source.cpp"
                        manifest_path = folder / f"case_{i:03d}_manifest.json"
                        stdout_path = folder / f"case_{i:03d}_original_stdout.txt"
                        with source_path.open("x", encoding="utf-8") as target:
                            target.write(emitted.cpp)
                        with manifest_path.open("x", encoding="utf-8") as target:
                            json.dump(emitted.manifest, target, indent=2, sort_keys=True)
                            target.write("\n")
                        rec["source_file"] = source_path.name
                        rec["source_manifest_file"] = manifest_path.name
                        rec["original_stdout_file"] = stdout_path.name
                    else:
                        stdout_path = None
                    from fpgai_research.validation.r2h_allocated_fanout_cpp import (
                        verify_allocated_gcc)
                    receipt = verify_allocated_gcc(
                        arch, emitted, runtime_input=inp,
                        original_trace_output=stdout_path)
                except (SourceProvenanceError, FileExistsError):
                    # Immutable original source-provenance conflicts and
                    # attempted overwrites must abort, not be relabelled
                    # as an ordinary numerical experiment failure.
                    raise
                except (ValueError, RuntimeError, AssertionError,
                        OSError, TimeoutError, subprocess.TimeoutExpired) as exc:
                    rec["status"] = "FAILED_E1_SOURCE_EXECUTION"
                    rec["reason"] = f"{type(exc).__name__}: {str(exc)[:350]}"
                else:
                    # Provenance errors are fatal: cannot demote to an ordinary
                    # execution failure or silently call it E1 certified.
                    if (receipt.get("source_cpp_sha256") != cert["cpp_sha256"]
                            or receipt.get("architecture_sha256")
                                != cert["architecture_sha256"]
                            or receipt.get("functional_sha256")
                                != cert["functional_sha256"]
                            or receipt.get("source_slot_map_sha256")
                                != cert["source_slot_map_sha256"]
                            or receipt.get("source_layout") != cert["layout"]):
                        raise SourceProvenanceError(
                            "R2o independent GCC source/certificate identity mismatch")
                    if stdout_path is not None and (
                            not stdout_path.is_file()
                            or sha256(stdout_path.read_bytes()).hexdigest()
                                != receipt.get("original_stdout_sha256")):
                        raise SourceProvenanceError(
                            "R2o source stdout SHA lineage mismatch")
                    if (receipt.get("status")
                            != "R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                            or receipt.get("original_GCC_compilations") != 1
                            or receipt.get("original_GCC_executions") != 1):
                        rec["status"] = "FAILED_E1_SOURCE_EXECUTION"
                        rec["reason"] = "Incomplete original GCC receipt"
                    else:
                        rec["status"] = "E1_GCC_MULTI_READER_QUALIFIED"
                        rec["e1_qualified"] = True
                        rec["gcc_receipt"] = {
                            k: receipt[k] for k in (
                                "source_cpp_sha256",
                                "original_input_sha256",
                                "original_stdout_sha256",
                                "original_runtime_source_event_sha256",
                                "original_actual_address_witness_count",
                                "source_scratch_unit_slots",
                                "source_all_declared_mutable_array_elements",
                                "original_all_intermediate_numeric_max_absolute_error",
                                "original_final_numeric_max_absolute_error",
                            )
                        }
        rows.append(rec)
    return {
        "schema": "fpgai-research/cst-R2o-source-qualified-multi-reader-mask/v1",
        "parent_state_sha256": fanout_state_hash(state),
        "fanout_readers": len([op for op in state.model.operators
                              if op.name.startswith("reader")]),
        "candidate_count": len(rows),
        "statuses": tuple(x["status"] for x in rows),
        "e0_certified": tuple(x["e0_certified"] for x in rows),
        "e1_qualified": tuple(x["e1_qualified"] for x in rows),
        "gcc_e1_attempts": attempted,
        "original_vitis_calls": 0, "original_vivado_calls": 0,
        "rl_policies_trained": 0, "hardware_feasibility_checked": False,
        "limitations": [
            "Only exact bounded MatMul 2..4-reader Add-chain Relu",
            "Only independently mapped reader PE width and scratch layout",
            "GCC host float for one input, no native fixed point or HLS cycles",
            "No trained RL, general DAG or hardware clock/BRAM validity",
        ],
        "rows": rows,
    }
