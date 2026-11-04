"""R2p shared C3 facade: original compilers are the *only* legal oracles.

Tests ensure no E0/E1 status promotion, actual compiler transitions, source
lineage, policy-feature privacy, real GCC host numerical execution and
unaltered frozen R2n/R2o API compatibility. No native/Vivado or RL calls.
"""
from __future__ import annotations

from dataclasses import replace
import shutil

import numpy as np
import pytest

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.decision_interface import (
    DecisionAction, create_state, source_executed_action_mask,
)
from fpgai_research.model_ir.fanout_decision_interface import (
    FanoutAccepted, create_fanout_state, source_qualified_multi_reader_mask,
)
from fpgai_research.model_ir import unified_decision_interface as api


def chain():
    return create_state(
        make_chain(3, (3,5,4,2)), exchange_policy="row_chain")


def fanout(n=2):
    return create_fanout_state(
        sample_fanout(n, m=2,k=2,n=3,q=4), layout="disjoint")


def chain_actions():
    return (
        DecisionAction("operator","mm0","pes",2),
        DecisionAction("operator","mm0","pes",0),
        DecisionAction("exchange","model","exchange_policy","whole_tensor"),
    )


def fanout_actions():
    return (
        DecisionAction("operator","reader0","pes",2),
        DecisionAction("operator","producer","pes",2),
        DecisionAction("memory","model","layout","per_element_optimal"),
    )


@pytest.mark.parametrize("domain", ("sequential","multi_reader_fanout"))
def test_unified_e0_must_reuse_real_compiler_transition_exactly(domain):
    state = chain() if domain=="sequential" else fanout()
    actions = chain_actions() if domain=="sequential" else fanout_actions()
    original_sha = api.unified_state_hash(state)
    observed = api.unified_action_mask(state, actions)
    assert observed["domain"] == domain
    assert observed["candidate_count"] == 3
    assert observed["parent_state_sha256"] == original_sha
    assert observed["original_gcc_calls"] == 0
    assert observed["native_vitis_calls"] == 0
    assert observed["original_vivado_calls"] == 0
    assert observed["rl_policies_trained"] == 0
    assert observed["statuses"] == (
        ("BOUNDED_CERTIFIED_E0", "REJECTED", "UNKNOWN_E0")
        if domain=="sequential" else
        ("BOUNDED_CERTIFIED_E0", "REJECTED", "BOUNDED_CERTIFIED_E0")
    )
    assert observed["source_admissible"] == (True,False,True)
    assert observed["e0_certified"] == (
        (True,False,False) if domain=="sequential" else (True,False,True)
    )
    assert observed["rows"][1]["cpp_sha256"] is None
    assert api.unified_state_hash(state) == original_sha
    for action, row in zip(actions, observed["rows"]):
        single = api.unified_transition(state, action)
        assert single.as_receipt() == row
        assert single.original_state is state
        if row["source_admissible"]:
            assert single.next_state is not state
            assert row["state_sha256"] == api.unified_state_hash(single.next_state)
            assert len(row["cpp_sha256"]) == 64
            assert len(row["certificate_sha256"]) == 64
        else:
            assert single.next_state is state


def test_true_one_transition_per_candidate_and_no_proxy(monkeypatch):
    state = fanout()
    originals = api.fanout_transition
    calls = []
    def counted(s, a):
        calls.append(a)
        return originals(s,a)
    monkeypatch.setattr(api, "fanout_transition", counted)
    actions = fanout_actions()
    out = api.unified_action_mask(state, actions)
    assert calls == list(actions)
    assert out["statuses"][1] == "REJECTED"


def test_unknown_e0_source_accepted_is_not_masked_as_certified():
    state=chain()
    result=api.unified_transition(
        state, DecisionAction("exchange","model","exchange_policy","whole_tensor"))
    assert result.source_admissible is True
    assert result.e0_certified is False
    assert result.status=="UNKNOWN_E0"
    assert result.reason
    assert result.state_sha256 != api.unified_state_hash(state)


@pytest.mark.parametrize("domain", ("sequential","multi_reader_fanout"))
def test_public_common_observation_is_weight_private_and_typed(domain):
    state = chain() if domain=="sequential" else fanout(3)
    graph = api.unified_graph_observation(state)
    assert graph["domain"] == domain
    assert graph["schema"].endswith("graph/v1")
    assert graph["nodes"] and graph["edges"]
    assert len([n for n in graph["nodes"] if n["type"]=="operator"])==len(
        state.model.operators)
    assert graph["target_limits"]
    assert not any("data" in n or "weights" in n for n in graph["nodes"])
    assert "full_source_state_sha256" not in str(graph)
    assert "source_state_sha256" not in str(graph)
    assert not any("reward" in n for n in graph["nodes"])
    assert all(
        "rank" in n and "reader_count" in n and n["reader_count"]>=0
        for n in graph["nodes"] if n["type"]=="tensor"
    )
    assert all(
        "phi_spatial_probe" in n and "mapping" in n
        for n in graph["nodes"] if n["type"]=="operator"
    )
    assert graph["source_domain"]
    assert api.unified_state_hash(state) not in str(graph)


@pytest.mark.skipif(shutil.which("g++") is None, reason="actual GCC required")
@pytest.mark.parametrize("domain", ("sequential","multi_reader_fanout"))
def test_real_original_gcc_e1_statuses_through_same_common_facade(domain):
    state=chain() if domain=="sequential" else fanout()
    actions=chain_actions() if domain=="sequential" else fanout_actions()
    x=(np.linspace(-.125,.125,9,dtype=np.float32).reshape(3,3)
       if domain=="sequential"
       else np.linspace(-.1,.1,4,dtype=np.float32).reshape(2,2))
    parent=api.unified_state_hash(state)
    observed=api.unified_source_mask(
        state,actions,runtime_input=x,max_e1_executions=3)
    assert observed["statuses"] == (
        ("E1_GCC_QUALIFIED","REJECTED","UNKNOWN_E0")
        if domain=="sequential" else
        ("E1_GCC_QUALIFIED","REJECTED","E1_GCC_QUALIFIED")
    )
    assert observed["e1_qualified"] == (
        (True,False,False) if domain=="sequential" else (True,False,True)
    )
    assert observed["e0_certified"] == (
        (True,False,False) if domain=="sequential" else (True,False,True)
    )
    assert observed["gcc_e1_attempts"] == (
        1 if domain=="sequential" else 2
    )
    assert observed["native_vitis_calls"] == 0
    assert observed["original_vivado_calls"] == 0
    assert observed["rl_policies_trained"] == 0
    assert observed["physical_feasibility_checked"] is False
    for row in observed["rows"]:
        if row["e1_gcc_qualified"]:
            assert row["gcc_receipt"]
            assert len(row["cpp_sha256"]) == 64
        else:
            assert row["gcc_receipt"] is None
    assert api.unified_state_hash(state) == parent


def test_invalid_typed_interface_and_action_budget_fail_closed():
    state=fanout()
    with pytest.raises(TypeError, match="existing typed bounded"):
        api.unified_state_hash(object())
    with pytest.raises(TypeError, match="typed DecisionAction"):
        api.unified_transition(state,object())
    with pytest.raises(TypeError, match="typed action candidates"):
        api.unified_action_mask(state,list(fanout_actions()))
    with pytest.raises(TypeError, match="typed action candidates"):
        api.unified_action_mask(state,fanout_actions()*3)
    with pytest.raises(ValueError, match="shared E1 budget"):
        api.unified_source_mask(
            state,fanout_actions(),runtime_input=np.zeros((2,2),np.float32),
            max_e1_executions=2)
    with pytest.raises(ValueError, match="shared E1 budget"):
        api.unified_source_mask(
            state,fanout_actions(),runtime_input=np.zeros((2,2),np.float32),
            max_e1_executions=True)


def test_tampered_original_compiler_sha_fails_instead_of_promoting_e0(monkeypatch):
    state=fanout()
    action=fanout_actions()[0]
    original = api.fanout_transition(state,action)
    assert isinstance(original,FanoutAccepted)
    forged = FanoutAccepted(
        original.state,{**original.certificate,"cpp_sha256":"not-a-digest"})
    monkeypatch.setattr(api,"fanout_transition",lambda s,a:forged)
    with pytest.raises(ValueError, match="source lineage"):
        api.unified_transition(state,action)


def test_forged_old_compiler_e1_boolean_cannot_be_promoted(monkeypatch):
    state=fanout()
    actions=fanout_actions()[:1]
    original = api.source_qualified_multi_reader_mask
    def falsify(s, a, **kwargs):
        raw=original(s,a,max_e1_executions=0,
                     runtime_input=kwargs["runtime_input"])
        raw["rows"][0]["e1_qualified"]=True
        return raw
    monkeypatch.setattr(api,"source_qualified_multi_reader_mask",falsify)
    with pytest.raises(ValueError, match="cannot promote"):
        api.unified_source_mask(
            state,actions,runtime_input=np.zeros((2,2),dtype=np.float32),
            max_e1_executions=1)


def test_no_fallback_between_distinct_supported_subdomains():
    state=fanout()
    old=api.unified_state_hash(state)
    wrong=DecisionAction("exchange","model","exchange_policy","row_chain")
    result=api.unified_transition(state,wrong)
    assert result.status=="REJECTED"
    assert result.original_state is state
    assert api.unified_state_hash(state)==old
