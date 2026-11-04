"""R3f six variable-size graph families × eight actions: source-only C2."""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_r3f_c2_variable_graph_coverage as sut


@pytest.mark.parametrize("name,parameter", sut.COHORTS)
def test_all_source_actions_replayed_through_same_compiler(name, parameter):
    row = sut.evaluate(name, parameter)
    assert row["candidate_count"] == 8
    assert len(row["statuses"]) == 8
    assert len(row["candidate_actions"]) == 8
    assert len(row["source_cpp_sha256"]) == 8
    assert row["certified"] + row["rejected"] + row["source_admitted_but_e0_unknown"] == 8
    assert row["mask_compiler_transitions"] == 8
    assert row["blind_replay_compiler_transitions"] == 8
    assert len(row["source_state_sha256"]) == 64
    assert len(row["graph_feature_digest"]) == 64
    if name.startswith("chain"):
        assert row["operator_count"] == parameter
    else:
        assert row["operator_count"] == 2 * parameter + 1
    if name.startswith("chain"):
        assert row["statuses"][1] == "REJECTED"
        assert row["statuses"][2] == "UNKNOWN_E0"
        assert row["statuses"][5] == "REJECTED"
        assert row["statuses"][7] == "REJECTED"
    else:
        assert row["statuses"][0] == "BOUNDED_CERTIFIED_E0"
        assert row["statuses"][4] == "BOUNDED_CERTIFIED_E0"
        for i in (2,3,5,6,7):
            assert row["statuses"][i] == "REJECTED"


def test_complete_six_graph_48_action_denominator():
    report = sut.run()
    assert report["schema"] == sut.SCHEMA
    assert report["prespecified_graphs"] == 6
    assert report["prespecified_candidates_per_graph"] == 8
    assert report["total_candidate_population"] == 48
    assert report["mask_compiler_transition_calls"] == 48
    assert report["blind_replay_compiler_transition_calls"] == 48
    assert report["total_e0_certified"] + report["total_source_admitted_but_e0_unknown"] + report["total_rejected"] == 48
    assert report["native_vitis_calls"] == 0
    assert report["source_gcc_execution_calls"] == 0
    assert report["original_vivado_calls"] == 0
    assert report["trained_agents"] == 0


def test_unregistered_graph_and_invalid_action_catalog_rejected():
    with pytest.raises(ValueError, match="undeclared"):
        sut.evaluate("cnn_unsupported", 9)


def _receipt():
    rows=[]
    for name, parameter in sut.COHORTS:
        statuses=(["BOUNDED_CERTIFIED_E0","REJECTED","UNKNOWN_E0",
                   "BOUNDED_CERTIFIED_E0","REJECTED","REJECTED",
                   "BOUNDED_CERTIFIED_E0","REJECTED"]
                  if name.startswith("chain_") else
                  ["BOUNDED_CERTIFIED_E0","BOUNDED_CERTIFIED_E0","REJECTED",
                   "REJECTED","BOUNDED_CERTIFIED_E0","REJECTED",
                   "REJECTED","REJECTED"])
        rows.append({
            "cohort": name, "graph_parameter": parameter,
            "candidate_count": 8, "statuses": statuses,
            "candidate_actions": [{"a": i} for i in range(8)],
            "mask_compiler_transitions": 8,
            "blind_replay_compiler_transitions": 8,
            "graph_feature_digest": "a" * 64,
            "certified": statuses.count("BOUNDED_CERTIFIED_E0"),
            "source_admitted_but_e0_unknown": statuses.count("UNKNOWN_E0"),
            "rejected": statuses.count("REJECTED"),
        })
    return {
        "schema": sut.SCHEMA,
        "prespecified_graphs": 6, "prespecified_candidates_per_graph": 8,
        "total_candidate_population": 48,
        "total_e0_certified": sum(r["certified"] for r in rows),
        "total_source_admitted_but_e0_unknown": sum(
            r["source_admitted_but_e0_unknown"] for r in rows),
        "total_rejected": sum(r["rejected"] for r in rows),
        "mask_compiler_transition_calls": 48,
        "blind_replay_compiler_transition_calls": 48,
        "native_vitis_calls": 0, "source_gcc_execution_calls": 0,
        "original_vivado_calls": 0, "trained_agents": 0,
        "rows": rows,
    }


def test_read_only_saved_action_audit_with_create_only_output(tmp_path):
    result = _receipt()
    dest = tmp_path / "r3f"
    sut.create_only(dest, result, revision="a" * 40)
    with pytest.raises(FileExistsError, match="already exists"):
        sut.create_only(dest, result, revision="a" * 40)
    data = sut.audit(dest)
    assert data["graph_population"] == 6
    assert data["candidate_population"] == 48
    assert data["certified"] == result["total_e0_certified"]
    assert data["new_compiler_or_vendor_calls"] == 0
    assert len(data["read_only_receipt_sha256"]) == 64


@pytest.mark.parametrize("change", [
    lambda a: a.update(total_candidate_population=47),
    lambda a: a["rows"][0]["statuses"].pop(),
    lambda a: a["rows"][3].update(graph_feature_digest="invalid"),
    lambda a: a["rows"][4].update(rejected=100),
    lambda a: a.update(trained_agents=2),
    lambda a: a.update(mask_compiler_transition_calls=0),
])
def test_read_only_audit_rejects_missing_or_corrupt_action_results(tmp_path, change):
    target = tmp_path / "r3f"
    sut.create_only(target, _receipt(), revision="b" * 40)
    file = target / "r3f_c2_source_coverage.json"
    data = json.loads(file.read_text())
    change(data)
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="R3f:"):
        sut.audit(target)



@pytest.mark.parametrize("readers", (2, 3, 4))
def test_same_fanout_model_and_same_action_change_legality_only_by_target_limit(readers):
    """C2 causal control: same source graph/action, different explicit limit.

    This is *target feasibility at source compiler level*, not a physical
    FPGA resource/timing proof or a Phi-conditioned value-lifetime theorem.
    """
    from fpgai_research.architecture.resolver import TargetLimits
    from fpgai_research.experiments.cst_fanout_models import sample_fanout
    from fpgai_research.model_ir.decision_interface import DecisionAction
    from fpgai_research.model_ir.fanout_decision_interface import create_fanout_state
    from fpgai_research.model_ir.unified_decision_interface import (
        unified_action_mask, unified_state_hash,
    )
    model = sample_fanout(readers, m=2,k=2,n=3,q=4)
    strict = create_fanout_state(
        model, layout="disjoint", limits=TargetLimits(max_parallel_units=1))
    relaxed = create_fanout_state(
        model, layout="disjoint", limits=TargetLimits(max_parallel_units=4))
    identical = (DecisionAction("operator", "reader0", "pes", 2),)
    old_strict = unified_state_hash(strict)
    old_relaxed = unified_state_hash(relaxed)
    strict_mask = unified_action_mask(strict, identical)
    relaxed_mask = unified_action_mask(relaxed, identical)
    assert strict_mask["statuses"] == ("REJECTED",)
    assert relaxed_mask["statuses"] == ("BOUNDED_CERTIFIED_E0",)
    assert strict_mask["e0_certified"] == (False,)
    assert relaxed_mask["e0_certified"] == (True,)
    assert unified_state_hash(strict) == old_strict
    assert unified_state_hash(relaxed) == old_relaxed
    assert strict_mask["native_vitis_calls"] == 0
    assert relaxed_mask["native_vitis_calls"] == 0


@pytest.mark.parametrize("depth", (2, 3, 4))
def test_true_compiler_state_renaming_and_weight_privacy_across_graph_sizes(depth):
    """C2 stronger invariance: recompile actual renamed and changed-weight models.

    Previous feature-list relabeling alone cannot establish that the public
    observation is invariant across distinct source-program identities.
    The graph shape and architecture remain fixed in these interventions.
    """
    from dataclasses import replace
    from fpgai_research.model_ir.contracts import FunctionalModel
    from fpgai_research.model_ir.decision_interface import (
        DecisionAction, create_state,
    )
    from fpgai_research.model_ir.unified_decision_interface import (
        unified_action_mask, unified_graph_observation, unified_state_hash,
    )
    from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
        structural_signature,
    )

    baseline, _catalog = sut._case(f"chain_d{depth}", depth)
    source = baseline.model
    observation = unified_graph_observation(baseline)
    source_sha = unified_state_hash(baseline)

    # Rename ALL actual source model tensor/operator identities consistently.
    names = {t.name: "renamed_" + t.name for t in source.tensors}
    renamed_source = FunctionalModel(
        tensors=tuple(replace(t, name=names[t.name]) for t in source.tensors),
        operators=tuple(replace(op, name="other_"+op.name,
            inputs=tuple(names[n] for n in op.inputs),
            outputs=tuple(names[n] for n in op.outputs))
            for op in source.operators),
        inputs=tuple(names[n] for n in source.inputs),
        outputs=tuple(names[n] for n in source.outputs),
    )
    renamed = create_state(renamed_source, exchange_policy="row_chain")
    assert unified_state_hash(renamed) != source_sha
    assert unified_graph_observation(renamed) == observation
    assert (structural_signature(unified_graph_observation(renamed)) ==
            structural_signature(observation))
    changed_action = DecisionAction("operator","other_mm0","pes",2)
    initial_action = DecisionAction("operator","mm0","pes",2)
    assert (unified_action_mask(baseline,(initial_action,))["statuses"] ==
            unified_action_mask(renamed,(changed_action,))["statuses"])

    # Change source weight DATA, but keep datatype, constant-ness,
    # operator/tensor shapes, architecture and policy identical.
    tensor = next(t for t in source.tensors if t.data is not None)
    altered_payload = (tensor.data[0]+0.25,)+tensor.data[1:]
    changed_source = replace(
        source, tensors=tuple(
            replace(t, data=altered_payload) if t.name==tensor.name else t
            for t in source.tensors))
    changed = create_state(changed_source, exchange_policy="row_chain")
    assert unified_state_hash(changed) != source_sha
    assert unified_graph_observation(changed) == observation
    assert (unified_action_mask(baseline,(initial_action,))["statuses"] ==
            unified_action_mask(changed,(initial_action,))["statuses"])
