"""R2o: independent multi-reader fanout decisions, E0/E1 controls (no vendor)."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import TargetLimits
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.decision_interface import DecisionAction
from fpgai_research.model_ir.fanout_decision_interface import (
    FanoutAccepted, FanoutRejected, create_fanout_state,
    fanout_certificate, fanout_state_hash, fanout_transition,
    multi_reader_graph_observation, source_qualified_multi_reader_mask,
)


def _state(readers=2, *, layout="disjoint", cap=None):
    return create_fanout_state(
        sample_fanout(readers, m=2, k=2, n=3, q=4),
        layout=layout,
        limits=TargetLimits(max_parallel_units=cap))


def _actions(readers=2):
    return (
        DecisionAction("operator", "reader0", "pes", 2),
        DecisionAction("operator", f"reader{readers-1}", "pes", 2),
        DecisionAction("operator", "producer", "pes", 2),
        DecisionAction("memory", "model", "layout", "per_element_optimal"),
    )


@pytest.mark.parametrize("readers", (2, 3, 4))
def test_true_multi_reader_phi_decisions_and_exact_source_certificates(readers):
    base = _state(readers)
    initial = fanout_certificate(base)
    original = fanout_state_hash(base)
    observed = multi_reader_graph_observation(base)
    assert observed["fanout_readers"] == readers
    assert not any("data" in node for node in observed["nodes"])
    assert sum(n["kind"] == "MatMul" for n in observed["nodes"]
               if n["type"] == "operator") == readers+1
    actions = _actions(readers)
    for action in (actions[0], actions[1], actions[3]):
        result = fanout_transition(base, action)
        assert isinstance(result, FanoutAccepted), result
        assert result.certificate["parent_state_sha256"] == original
        assert result.certificate["fanout_readers"] == readers
        assert result.certificate["source_event_count"] > 0
        assert result.certificate["status"] == "E0_SOURCE_EMITTED_AND_MULTI_READER_CERTIFIED"
        assert result.certificate["cpp_sha256"] != initial["cpp_sha256"]
        assert len(result.certificate["certificate_sha256"]) == 64
        assert result.certificate["native_vitis_calls"] == 0
        assert result.certificate["physical_feasibility_checked"] is False
    last = fanout_transition(base, actions[1])
    assert isinstance(last, FanoutAccepted)
    assert last.state.requests[readers][1].pes == 2
    assert base.requests[readers][1].pes == 1
    assert fanout_state_hash(base) == original
    for action in (actions[2],
                   DecisionAction("operator", "reader0", "pes", 0),
                   DecisionAction("operator", "reader0", "simd", 2),
                   DecisionAction("memory", "model", "layout", "invented"),
                   DecisionAction("memory", "model", "layout", "disjoint")):
        r = fanout_transition(base, action)
        assert isinstance(r, FanoutRejected)
        assert r.state is base
    optimized = fanout_transition(base, actions[3])
    assert optimized.certificate["source_scratch_unit_slots"] == (
        optimized.certificate["fixed_order_minimum_unit_slots"])
    assert optimized.certificate["source_scratch_unit_slots"] < (
        initial["source_scratch_unit_slots"])


def test_target_limit_and_bool_edit_refused():
    state = _state(cap=1)
    assert isinstance(
        fanout_transition(
            state, DecisionAction("operator", "reader0", "pes", 2)),
        FanoutRejected)
    assert isinstance(
        fanout_transition(
            _state(), DecisionAction("operator", "reader0", "pes", True)),
        FanoutRejected)


@pytest.mark.skipif(shutil.which("g++") is None, reason="real host GCC is required")
def test_real_gcc_multi_reader_source_address_and_numeric_E1():
    state = _state(2)
    actions = _actions(2)
    x = np.linspace(-.1, .1, 4, dtype=np.float32).reshape(2, 2)
    outcome = source_qualified_multi_reader_mask(
        state, actions, runtime_input=x, max_e1_executions=2)
    assert outcome["candidate_count"] == 4
    assert outcome["statuses"] == (
        "E1_GCC_MULTI_READER_QUALIFIED",
        "E1_GCC_MULTI_READER_QUALIFIED",
        "REJECTED", "UNKNOWN_E1_NOT_RUN")
    assert outcome["e0_certified"] == (True, True, False, True)
    assert outcome["e1_qualified"] == (True, True, False, False)
    assert outcome["gcc_e1_attempts"] == 2
    assert outcome["original_vitis_calls"] == 0
    assert outcome["original_vivado_calls"] == 0
    assert outcome["rl_policies_trained"] == 0
    for row in outcome["rows"][:2]:
        receipt = row["gcc_receipt"]
        assert receipt["original_actual_address_witness_count"] > 0
        assert len(receipt["original_stdout_sha256"]) == 64
        assert receipt["original_input_sha256"] == sha256(x.tobytes()).hexdigest()
        assert receipt["original_all_intermediate_numeric_max_absolute_error"] < 1e-4
        assert receipt["original_final_numeric_max_absolute_error"] < 1e-4


def test_zero_e1_budget_never_promotes_source_e0_to_actual_gcc():
    state = _state()
    a = _actions()[:1]
    x = np.zeros((2, 2), dtype=np.float32)
    out = source_qualified_multi_reader_mask(
        state, a, runtime_input=x, max_e1_executions=0)
    assert out["statuses"] == ("UNKNOWN_E1_NOT_RUN",)
    assert out["e0_certified"] == (True,)
    assert out["e1_qualified"] == (False,)
    assert out["gcc_e1_attempts"] == 0


def test_bad_runtime_input_and_typed_action_bounds_fail_closed():
    state = _state()
    x = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="fixed input"):
        source_qualified_multi_reader_mask(
            state, _actions()[:1],
            runtime_input=np.zeros((4,), dtype=np.float32))
    with pytest.raises(ValueError, match="fixed input"):
        source_qualified_multi_reader_mask(
            state, _actions()[:1],
            runtime_input=np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="budget"):
        source_qualified_multi_reader_mask(
            state, _actions()[:1], runtime_input=x,
            max_e1_executions=-1)
    with pytest.raises(TypeError, match="tuple"):
        source_qualified_multi_reader_mask(
            state, list(_actions()), runtime_input=x)
    with pytest.raises(ValueError, match="candidate action budget"):
        source_qualified_multi_reader_mask(
            state, (_actions()[:1])*33, runtime_input=x,
            max_e1_executions=0)
    with pytest.raises(ValueError, match="unknown operator"):
        create_fanout_state(
            state.model, overrides={"ghost": ArchitectureRequest(pes=2)})


def test_gcc_failure_preserved_without_false_source_qualification(monkeypatch):
    import fpgai_research.validation.r2h_allocated_fanout_cpp as gcc
    def fail(*a, **k):
        raise RuntimeError("injected actual source failure")
    monkeypatch.setattr(gcc, "verify_allocated_gcc", fail)
    state = _state()
    out = source_qualified_multi_reader_mask(
        state, _actions()[:1], runtime_input=np.zeros((2, 2), dtype=np.float32))
    assert out["statuses"] == ("FAILED_E1_SOURCE_EXECUTION",)
    assert out["e0_certified"] == (True,)
    assert out["e1_qualified"] == (False,)
    assert "injected actual source failure" in out["rows"][0]["reason"]


def test_source_provenance_tampering_is_fatal_not_ordinary_failure(monkeypatch):
    import fpgai_research.validation.r2h_allocated_fanout_cpp as gcc
    def forged(*a, **k):
        return {"status":"R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED",
                "source_cpp_sha256":"0"*64}
    monkeypatch.setattr(gcc, "verify_allocated_gcc", forged)
    state = _state()
    with pytest.raises(ValueError, match="identity mismatch"):
        source_qualified_multi_reader_mask(
            state, _actions()[:1],
            runtime_input=np.zeros((2, 2), dtype=np.float32))


def test_independently_mapped_reader_ownership_changes_e0_phi():
    baseline = _state(2)
    a = fanout_transition(
        baseline, DecisionAction("operator", "reader0", "pes", 2))
    b = fanout_transition(
        baseline, DecisionAction("operator", "reader1", "pes", 2))
    assert isinstance(a, FanoutAccepted)
    assert isinstance(b, FanoutAccepted)
    assert a.certificate["architecture_sha256"] != b.certificate["architecture_sha256"]
    assert a.certificate["state_sha256"] != b.certificate["state_sha256"]
    assert multi_reader_graph_observation(a.state) != multi_reader_graph_observation(b.state)
