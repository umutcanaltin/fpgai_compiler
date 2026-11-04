"""Paper C3 uses policy-independent legality, NOT measured RL quality."""
from __future__ import annotations

import csv
import json

import pytest

from fpgai_research.experiments.cst_rl_interface_study import (
    CATALOG, run, _anonymous_rename, _visible_features,
)
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.decision_interface import (
    create_state, graph_observation, state_hash,
)


def test_r3_r4_original_bounded_rl_compatibility_campaign(tmp_path):
    output=tmp_path/"study"
    report=run(output=output)
    assert report["graphs"]==3
    assert report["variable_operator_depths"]==[2,3,4]
    assert report["renaming_invariance_cases"]==3
    assert report["identical_proposer_certificate_cases"]==3
    assert report["certified_action_proposals"]==9
    assert report["certified_legal_actions"]==3
    assert report["source_admissible_but_unknown_actions"]==3
    assert report["native_vitis_calls"]==0
    assert report["rl_policies_trained"]==0
    assert report["rl_sample_efficiency_measured"] is False
    assert report["external_peer_parity_established"] is False
    assert json.loads((output/"summary.json").read_text())==report
    with (output/"cases.csv").open(newline="") as file:
        cases=list(csv.DictReader(file))
    assert len(cases)==3
    assert [int(c["graph_depth"]) for c in cases]==[2,3,4]
    assert all(c["source_legal_but_certificate_unknown_actions"]=="1"
               for c in cases)
    with pytest.raises(FileExistsError,match="create-only"):
        run(output=output)


def test_graph_renaming_does_not_hijack_full_source_identity():
    _,rows,widths=CATALOG[1]
    model=make_chain(rows,widths)
    original=create_state(model,exchange_policy="row_chain")
    renamed=create_state(_anonymous_rename(model),exchange_policy="row_chain")
    assert state_hash(original)!=state_hash(renamed)
    a,b=graph_observation(original),graph_observation(renamed)
    assert _visible_features(a)==_visible_features(b)
    assert a["full_source_state_sha256"]!=b["full_source_state_sha256"]


def test_raw_data_observations_are_rejected_by_interface_study():
    with pytest.raises(ValueError,match="Raw tensor payload leaked"):
        _visible_features({
            "observation_kind":"public-static-no-weights",
            "nodes":[{"id":"t0","type":"tensor","data":[0.2]}],
            "edges":[],"exchange_policy":"row_chain","row_block":1,
            "target_limits":{},"boundary_evidence_status":"UNKNOWN",
        })


def test_no_claim_of_rl_policy_reward_or_physical_qualification(tmp_path):
    report=run(output=tmp_path/"fresh")
    assert "rl_reward" not in report
    assert "trained_policy_return" not in report
    assert "physical_clock_ns" not in report
    assert report["evidence"]=="E0-certificate-backed-complete-design-action-protocol-only"
