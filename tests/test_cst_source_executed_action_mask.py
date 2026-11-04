"""C3 policy-neutral actual source E1 action mask; no native vendor calls."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.decision_interface import (
    DecisionAction, create_state, source_executed_action_mask,
)


def _fixture():
    state = create_state(
        make_chain(3, (3, 5, 4, 2)), exchange_policy="row_chain")
    x = np.linspace(-.125, .125, 9, dtype=np.float32).reshape(3, 3)
    actions = (
        DecisionAction("operator", "mm0", "pes", 2),
        DecisionAction("operator", "mm0", "pes", 0),
        DecisionAction("exchange", "model", "exchange_policy", "whole_tensor"),
    )
    return state, x, actions


@pytest.mark.skipif(shutil.which("g++") is None, reason="real GCC required")
def test_e1_action_mask_really_executes_original_cpp_with_provenance():
    state, x, actions = _fixture()
    observed = source_executed_action_mask(
        state, actions, runtime_input=x)
    assert observed["candidate_count"] == 3
    assert observed["source_admissible"] == (True, False, True)
    assert observed["e0_certified"] == (True, False, False)
    assert observed["e1_qualified"] == (True, False, False)
    assert observed["statuses"] == (
        "E1_GCC_SOURCE_QUALIFIED", "REJECTED",
        "UNKNOWN_E0_CERTIFICATE",
    )
    assert observed["e1_executions_attempted"] == 1
    assert observed["native_vitis_calls"] == 0
    assert observed["rl_policies_trained"] == 0
    assert observed["physical_feasibility_checked"] is False
    assert len(observed["rows"][0]["qualification_sha256"]) == 64
    assert observed["rows"][1]["state_sha256"] is None
    assert observed["rows"][2]["qualification_sha256"] is None
    # Input and original complete design have not been changed.
    assert state.requests[0][1].pes == 1
    assert x.shape == (3, 3)


def test_invalid_input_and_budget_fail_before_any_native_or_gcc(monkeypatch):
    import fpgai_research.model_ir.decision_interface as api
    state, x, actions = _fixture()
    monkeypatch.setattr(api, "qualify_accepted_source", lambda *a, **k:
                        pytest.fail("Should not execute GCC"))
    with pytest.raises(ValueError, match="budget"):
        source_executed_action_mask(
            state, actions, runtime_input=x, max_e1_executions=2)
    with pytest.raises(ValueError, match="Finite runtime input"):
        source_executed_action_mask(
            state, actions[:1], runtime_input=np.zeros((1, 9)))
    with pytest.raises(ValueError, match="Finite runtime input"):
        source_executed_action_mask(
            state, actions[:1], runtime_input=np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="Explicit fixed host input"):
        source_executed_action_mask(
            state, actions[:1], runtime_input=None)
    with pytest.raises(ValueError, match="budget"):
        source_executed_action_mask(
            state, actions[:1], runtime_input=x, max_e1_executions=0)
    with pytest.raises(TypeError, match="finite tuple"):
        source_executed_action_mask(
            state, list(actions), runtime_input=x)


def test_failed_gcc_execution_is_not_silently_accepted(monkeypatch):
    import fpgai_research.model_ir.decision_interface as api
    state, x, actions = _fixture()
    def fail(*a, **k):
        raise RuntimeError("deliberately failing C++ source")
    monkeypatch.setattr(api, "qualify_accepted_source", fail)
    observed = source_executed_action_mask(
        state, actions[:1], runtime_input=x)
    assert observed["statuses"] == ("FAILED_E1_SOURCE_EXECUTION",)
    assert observed["e1_qualified"] == (False,)
    assert observed["e0_certified"] == (True,)
    assert observed["e1_executions_attempted"] == 1
    assert "deliberately failing" in observed["rows"][0]["reason"]


def test_source_provenance_mismatch_is_fatal_not_an_unknown(monkeypatch):
    import fpgai_research.model_ir.decision_interface as api
    state, x, actions = _fixture()
    def tamper(*a, **k):
        return {
            "state_sha256": "0" * 64, "cpp_sha256": "1" * 64,
            "host_numerical": {"cpp_sha256": "1" * 64, "accepted": True},
            "source_value_lifetime": {
                "cpp_sha256": "1" * 64,
                "status": "SOURCE_VALUE_LIFETIMES_VERIFIED",
                "last_reader_releases_verified": 1,
            },
        }
    monkeypatch.setattr(api, "qualify_accepted_source", tamper)
    with pytest.raises(ValueError, match="source identity"):
        source_executed_action_mask(
            state, actions[:1], runtime_input=x)


def test_e1_status_cannot_be_claimed_from_partial_host_evidence(monkeypatch):
    import fpgai_research.model_ir.decision_interface as api
    state, x, actions = _fixture()
    def partial(accepted, **kwargs):
        return {
            "state_sha256": accepted.certificate["state_sha256"],
            "cpp_sha256": accepted.certificate["cpp_sha256"],
            "host_numerical": {
                "cpp_sha256": accepted.certificate["cpp_sha256"],
                "accepted": True,
            },
            "source_value_lifetime": {
                "cpp_sha256": accepted.certificate["cpp_sha256"],
                "status": "UNKNOWN",
            },
            "strongest_evidence": "E1-executed-GCC-partial",
            "qualification_sha256": "f" * 64,
        }
    monkeypatch.setattr(api, "qualify_accepted_source", partial)
    observed = source_executed_action_mask(
        state, actions[:1], runtime_input=x)
    assert observed["statuses"] == ("FAILED_E1_SOURCE_EXECUTION",)
    assert observed["e0_certified"] == (True,)
    assert observed["e1_qualified"] == (False,)
