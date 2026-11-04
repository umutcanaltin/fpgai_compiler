"""E0/E1 tests for a policy-neutral certified compiler decision interface.

The same action must be handled identically whether proposed by an RL
agent, human, heuristic or random search. No RL training or vendor calls.
"""
from __future__ import annotations

from dataclasses import replace, asdict
import json
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import TargetLimits, resolve
from fpgai_research.compiler.cst_pipeline import compile_cst
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import (
    ArchitectureRequest, FunctionalModel, Operator, Tensor)
from fpgai_research.model_ir.decision_interface import (
    Accepted, DecisionAction, Rejected, action_mask, candidate_actions,
    certificate, create_state, graph_observation, state_hash, transition,
    validate_state, qualify_accepted_source)
from fpgai_research.validation.host import validate_host


def study(rows=3, *, cap=None):
    return create_state(
        make_chain(rows, (3, 5, 4, 2)), exchange_policy="row_chain",
        limits=TargetLimits(max_parallel_units=cap))


def test_policy_neutral_actions_compile_same_source_and_certificate():
    state=study(3)
    action=DecisionAction("operator","mm0","pes",2)
    one=transition(state,action)
    two=transition(state,action)
    assert isinstance(one,Accepted)
    assert one.state.requests[0][1].pes == 2
    assert state.requests[0][1].pes == 1  # immutable previous state
    assert one.certificate == two.certificate
    assert one.certificate["parent_state_sha256"] == state_hash(state)
    assert one.certificate["state_sha256"] == state_hash(one.state)
    assert one.certificate["exchange_policy"] == "row_chain"
    assert one.certificate["schedule_verification_status"] == "VERIFIED"
    assert one.certificate["source_scratch_elements"] == 9
    assert one.certificate["native_vitis_calls"] == 0
    arch, compilation=validate_state(one.state)
    assert compilation.emission.manifest["cpp_sha256"] == one.certificate["cpp_sha256"]
    assert arch.points[0].effective.pes == 2


def test_sound_mask_on_bounded_complete_candidates():
    state=study(3,cap=2)
    proposals=(
        DecisionAction("operator","mm1","pes",2),
        DecisionAction("operator","mm1","pes",3),
        DecisionAction("operator","mm0","pes",0),
        DecisionAction("exchange","model","row_block",2),
        DecisionAction("exchange","model","row_block",4),
        DecisionAction("operator","missing","pes",2),
        DecisionAction("operator","mm0","precision_bits",99),
    )
    mask=action_mask(state,proposals)
    assert mask["legal"] == (True,False,False,True,False,False,False)
    assert not mask["native_hls_feasibility_checked"]
    for action, expected in zip(proposals, mask["legal"]):
        verdict=transition(state,action)
        assert isinstance(verdict,Accepted) == expected
        if not expected:
            assert isinstance(verdict,Rejected)
            assert verdict.state is state
            assert verdict.counterexample["field"] == action.field
            assert verdict.reason


def test_invalid_partial_and_unsupported_scopes_fail_closed():
    state=study()
    actions=(
        DecisionAction("operator","mm0","simd",True),
        DecisionAction("operator","mm0","tile",("bad",)),
        DecisionAction("exchange","model","intermediate_tensor",12),
        DecisionAction("global","model","reward",123),
        DecisionAction("exchange","mm0","row_block",2),
        DecisionAction("exchange","model","exchange_policy","stream"),
        DecisionAction("operator","mm0","pes",1),
    )
    assert all(isinstance(transition(state,a),Rejected) for a in actions)
    assert all(not b for b in action_mask(state,actions)["legal"])


def test_device_limit_changes_admissible_actions_without_fake_resource_prediction():
    allowed=transition(study(cap=4),DecisionAction("operator","mm0","pes",4))
    blocked=transition(study(cap=2),DecisionAction("operator","mm0","pes",4))
    assert isinstance(allowed,Accepted)
    assert isinstance(blocked,Rejected)
    assert "CAPACITY" in blocked.reason
    assert allowed.certificate["status"] == "SOURCE_EMITTED_AND_COMPILER_CONTRACT_ACCEPTED"
    assert "HLS physical feasibility" in json.dumps(allowed.certificate["limitations"])


def test_row_block_changes_actual_source_and_bounds():
    base=study(5)
    act=DecisionAction("exchange","model","row_block",2)
    result=transition(base,act)
    assert isinstance(result,Accepted)
    assert result.certificate["source_scratch_elements"] == 18
    assert result.certificate["cpp_sha256"] != certificate(base)["cpp_sha256"]
    assert result.certificate["functional_sha256"] == certificate(base)["functional_sha256"]
    assert isinstance(transition(base,DecisionAction(
        "exchange","model","row_block",6)),Rejected)


def test_public_graph_observation_is_weight_free_and_rename_equivariant():
    original=study()
    obs=graph_observation(original)
    assert obs["observation_kind"]=="public-static-no-weights"
    assert len([n for n in obs["nodes"] if n["type"]=="operator"])==3
    assert len([e for e in obs["edges"] if e["kind"]=="produces"])==3
    assert obs["boundary_evidence_status"]=="BOUNDED_PHI_SPATIAL_PROJECTIONS"
    assert sum("mapped_exchange" in n for n in obs["nodes"])==2
    assert not any("data" in n for n in obs["nodes"])
    assert all("phi_probe_projection" in n for n in obs["nodes"]
               if n["type"]=="operator")
    source=original.model
    names={t.name:"renamed_"+t.name for t in source.tensors}
    other=FunctionalModel(
        tensors=tuple(replace(t,name=names[t.name]) for t in source.tensors),
        operators=tuple(replace(op,name="r_"+op.name,
            inputs=tuple(names[n] for n in op.inputs),
            outputs=tuple(names[n] for n in op.outputs))
            for op in source.operators),
        inputs=tuple(names[n] for n in source.inputs),
        outputs=tuple(names[n] for n in source.outputs))
    renamed=create_state(other,exchange_policy="row_chain")
    obs2=graph_observation(renamed)
    assert obs["nodes"]==obs2["nodes"]
    assert obs["edges"]==obs2["edges"]
    assert obs["full_source_state_sha256"]!=obs2["full_source_state_sha256"]


def test_operator_phi_probe_reflects_real_mapping_updates():
    state=study()
    before=graph_observation(state)
    after=transition(state,DecisionAction("operator","mm0","pes",2))
    assert isinstance(after,Accepted)
    changed=graph_observation(after.state)
    def mapped(obs):
        return next(n for n in obs["nodes"] if n["id"]=="o0")["phi_probe_projection"]
    assert mapped(before) != mapped(changed)
    def boundary(obs):
        return next(n for n in obs["nodes"] if n["id"]=="t2")[
            "mapped_exchange"]["producer_phi_projection_sha256"]
    assert boundary(before)!=boundary(changed)


def test_finite_action_catalog_is_deterministic_and_bounded():
    state=study()
    first=candidate_actions(state,max_actions=13)
    second=candidate_actions(state,max_actions=13)
    assert first==second and len(first)==13
    assert all(isinstance(a,DecisionAction) for a in first)
    assert len(action_mask(state,first)["legal"])==13
    with pytest.raises(ValueError,match="budget"):
        candidate_actions(state,max_actions=0)


@pytest.mark.skipif(shutil.which("g++") is None,reason="g++ required")
def test_accepted_architecture_produces_correct_executed_cpp():
    state=study()
    proposal=transition(state,DecisionAction("operator","mm1","simd",2))
    assert isinstance(proposal,Accepted)
    _,compiled=validate_state(proposal.state)
    x=np.linspace(-.2,.2,9,dtype=np.float32).reshape(3,3)
    numerical=validate_host(state.model,compiled.emission,x,atol=1e-5,rtol=1e-4)
    assert numerical["accepted"],numerical
    assert numerical["cpp_sha256"]==proposal.certificate["cpp_sha256"]
    qualified=qualify_accepted_source(
        proposal, runtime_input=x, verify_value_trace=True)
    assert qualified["strongest_evidence"]=="E1-executed-GCC-value-and-numerical"
    assert qualified["source_value_lifetime"]["status"]=="SOURCE_VALUE_LIFETIMES_VERIFIED"
    assert qualified["host_numerical"]["accepted"]
    assert qualified["native_vitis_calls"]==0
    assert len(qualified["qualification_sha256"])==64
