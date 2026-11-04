"""C2 compiled-once certified source catalog regression tests; no trained RL."""
from __future__ import annotations

from dataclasses import replace

import pytest

from fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage import COHORTS, _case
from fpgai_research.model_ir import compiled_action_catalog as cat
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask, unified_state_hash, unified_transition,
)


@pytest.mark.parametrize("name,width", COHORTS)
def test_reuses_one_actual_compiler_transition_per_catalog_action(name, width):
    state, actions = _case(name, width)
    old = unified_state_hash(state)
    result = cat.build_catalog(state, actions)
    assert result.compiler_transitions == 8
    assert len(result.mask()) == len(result.statuses()) == 8
    assert result.as_receipt()["hardware_feasibility_checked"] is False
    assert result.as_receipt()["native_vitis_calls"] == 0
    assert result.as_receipt()["trained_agents"] == 0
    assert result.statuses() == unified_action_mask(state, actions)["statuses"]
    assert result.mask() == unified_action_mask(state, actions)["e0_certified"]
    assert unified_state_hash(state) == old
    first = next(i for i, flag in enumerate(result.mask()) if flag)
    decision = cat.select_certified(result, first, current_state=state)
    independently = unified_transition(state, actions[first])
    assert decision.as_receipt() == independently.as_receipt()
    assert decision.original_state is state
    assert decision.next_state is not state
    assert unified_state_hash(decision.next_state) == decision.state_sha256
    with pytest.raises(ValueError, match="different state object"):
        cat.select_certified(result, first, current_state=decision.next_state)
    for i, legal in enumerate(result.mask()):
        if not legal:
            with pytest.raises(ValueError, match="REJECTED or UNKNOWN"):
                cat.select_certified(result, i, current_state=state)


def test_cached_selection_has_zero_additional_compiler_calls(monkeypatch):
    state, actions = _case("chain_d3",3)
    calls = []
    original = cat.unified_transition
    def traced(s,a):
        calls.append(a)
        return original(s,a)
    monkeypatch.setattr(cat,"unified_transition",traced)
    built = cat.build_catalog(state, actions)
    assert calls == list(actions)
    monkeypatch.setattr(cat,"unified_transition",
                        lambda *_args: (_ for _ in ()).throw(
                            AssertionError("selection incorrectly recompiled")))
    idx = next(i for i,v in enumerate(built.mask()) if v)
    selected=cat.select_certified(built,idx,current_state=state)
    assert selected.status=="BOUNDED_CERTIFIED_E0"
    assert selected.state_sha256
    assert built.compiler_transitions==len(actions)


def test_type_action_count_and_source_binding_fail_closed():
    state, actions = _case("fanout_r2",2)
    with pytest.raises(TypeError,match="typed actions"):
        cat.build_catalog(state,list(actions))
    with pytest.raises(TypeError,match="typed actions"):
        cat.build_catalog(state,actions+(actions[0],))
    built=cat.build_catalog(state,actions)
    with pytest.raises(IndexError,match="outside"):
        cat.select_certified(built,-1,current_state=state)
    with pytest.raises(IndexError,match="outside"):
        cat.select_certified(built,True,current_state=state)
    with pytest.raises(TypeError,match="in-process"):
        cat.select_certified(None,0,current_state=state)
    false_decision=replace(built.decisions[0],cpp_sha256="broken")
    forged=replace(built,decisions=(false_decision,)+built.decisions[1:])
    with pytest.raises(ValueError,match="identity invalid"):
        cat.select_certified(forged,0,current_state=state)


def test_empty_catalog_is_bounded_and_select_is_impossible():
    state,_actions = _case("chain_d2",2)
    catalog=cat.build_catalog(state,())
    assert catalog.statuses()==()
    assert catalog.mask()==()
    assert catalog.compiler_transitions==0
    with pytest.raises(IndexError):
        cat.select_certified(catalog,0,current_state=state)
