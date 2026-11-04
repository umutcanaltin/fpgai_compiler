"""Bounded certificate-backed mask must distinguish SOURCE and UNKNOWN.

No RL policy, Gym training, vendor calls or inference about physical HLS.
"""
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.decision_interface import (
    DecisionAction, action_mask, certified_action_mask, create_state,
)

def test_certificate_mask_separates_source_admission_and_mapped_e0():
    model=make_chain(3,(3,5,4,2))
    state=create_state(model,exchange_policy="row_chain")
    acts=(
        DecisionAction("operator",model.operators[0].name,"pes",2),
        DecisionAction("operator",model.operators[0].name,"pes",0),
        DecisionAction("exchange","model","exchange_policy","whole_tensor"),
        DecisionAction("exchange","model","row_block",2),
    )
    mask=certified_action_mask(state,acts)
    assert mask["source_admissible"] == action_mask(state,acts)["legal"]
    assert mask["status"] == (
        "BOUNDED_MAPPED_VALUE","REJECTED","UNKNOWN","BOUNDED_MAPPED_VALUE")
    assert mask["legal"] == (True,False,False,True)
    from fpgai_research.model_ir.decision_interface import transition, Accepted
    accepted=transition(state,acts[0])
    assert isinstance(accepted,Accepted)
    assert accepted.certificate["mapped_logical_event_contract"]["status"] == "VERIFIED"
    assert accepted.certificate["mapped_logical_event_contract"]["events_checked"] > 0
    assert mask["native_hls_feasibility_checked"] is False

def test_certificate_mask_is_stable_and_never_marks_unknown_legal():
    graph=make_chain(3,(3,5,4,2))
    state=create_state(graph,exchange_policy="row_chain")
    actions=(DecisionAction("exchange","model","exchange_policy","whole_tensor"),)
    a=certified_action_mask(state,actions)
    assert a==certified_action_mask(state,actions)
    assert a["source_admissible"]==(True,)
    assert a["legal"]==(False,)
    assert a["status"]==("UNKNOWN",)
