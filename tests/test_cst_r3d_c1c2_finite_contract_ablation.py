"""R3d C1/C2 restricted oracle, gate-ablation and graph invariance tests.

No agent/Gym/Vitis/Vivado use. Existing source compiler is the ONLY actual
certification authority; the fixed six-case oracle is deliberately narrow.
"""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from fpgai_research.experiments import cst_r3d_c1c2_finite_contract_ablation as study
from fpgai_research.experiments.cst_r2p_unified_interface_study import _cases
from fpgai_research.model_ir.unified_decision_interface import unified_graph_observation


def test_real_compiler_restricted_finite_gate_ablation_keeps_full_denominator():
    report = study.run()
    assert report["schema"] == study.SCHEMA
    assert report["finite_candidate_denominator"] == 6
    assert report["e0_certified"] == 3
    assert report["source_admitted_but_e0_unknown"] == 1
    assert report["rejected"] == 2
    assert report["mask_construction_compiler_transitions"] == 6
    assert report["blind_feedback_compiler_transitions"] == 6
    assert report["rl_policies_trained"] == 0
    assert report["native_vitis_calls"] == 0
    assert report["original_gcc_calls"] == 0
    assert report["original_vivado_calls"] == 0
    assert len(report["rows"]) == 2
    assert [r["action_replay_statuses"] for r in report["rows"]] == [
        list(study.EXPECTED_RESTRICTED[0]),
        list(study.EXPECTED_RESTRICTED[1]),
    ]
    for row in report["rows"]:
        assert row["candidate_count"] == 3
        assert row["e0_unverified_not_promoted"] >= 0
        assert row["graph_signature"]["node_count"] > 0
        assert row["graph_signature"]["edge_count"] > 0
        assert len(row["graph_signature"]["digest"]) == 64
        assert row["mask_build_wall_ns"] >= 0
        assert row["blind_feedback_wall_ns"] >= 0


@pytest.mark.parametrize("case_index", (0, 1))
def test_public_graph_signature_invariant_to_node_edge_order_and_renaming(case_index):
    domain, state, _actions, _input = _cases()[case_index]
    graph = unified_graph_observation(state)
    base = study.structural_signature(graph)
    shuffled = deepcopy(graph)
    shuffled["nodes"].reverse()
    shuffled["edges"].reverse()
    rename = {n["id"]: f"anonymous_{i+101}" for i, n in enumerate(shuffled["nodes"])}
    for n in shuffled["nodes"]:
        n["id"] = rename[n["id"]]
    for e in shuffled["edges"]:
        e["src"] = rename[e["src"]]
        e["dst"] = rename[e["dst"]]
    assert study.structural_signature(shuffled) == base
    assert base["domain"] == domain
    assert "fpgai" in base["schema"]
    # An actual source-relevant typed feature edit must change the signature,
    # but digest inequality is not an independent functional semantics proof.
    changed = deepcopy(graph)
    tensor = next(n for n in changed["nodes"] if n["type"] == "tensor")
    tensor["shape"][0] += 1
    assert study.structural_signature(changed)["digest"] != base["digest"]


def test_invalid_dangling_and_duplicate_public_graphs_fail_closed():
    graph = unified_graph_observation(_cases()[0][1])
    bad = deepcopy(graph)
    bad["nodes"][0]["id"] = bad["nodes"][1]["id"]
    with pytest.raises(ValueError, match="duplicated"):
        study.structural_signature(bad)
    bad = deepcopy(graph)
    bad["edges"][0]["src"] = "not-an-existing-node"
    with pytest.raises(ValueError, match="dangling"):
        study.structural_signature(bad)
    with pytest.raises(ValueError, match="unsupported"):
        study.structural_signature({})


def test_mask_replay_mismatch_fails_instead_of_fabricating_certification(monkeypatch):
    domain, state, actions, _input = _cases()[0]
    bad = ("BOUNDED_CERTIFIED_E0", "BOUNDED_CERTIFIED_E0", "UNKNOWN_E0")
    with pytest.raises(ValueError, match="restricted oracle"):
        study.finite_ablation(domain=domain, state=state, actions=actions,
                              expected=bad)


def _minimal_record():
    rows = []
    for domain, statuses in zip(study.DOMAINS, study.EXPECTED_RESTRICTED):
        rows.append({
            "domain": domain,
            "action_replay_statuses": list(statuses),
            "graph_signature": {"digest": "c" * 64},
            "mask_build_compiler_transitions": 3,
            "blind_feedback_compiler_transitions": 3,
        })
    return {
        "schema": study.SCHEMA, "rows": rows,
        "finite_candidate_denominator": 6,
        "e0_certified": 3, "source_admitted_but_e0_unknown": 1,
        "rejected": 2,
        "mask_construction_compiler_transitions": 6,
        "blind_feedback_compiler_transitions": 6,
        "native_vitis_calls": 0, "original_gcc_calls": 0,
        "original_vivado_calls": 0, "rl_policies_trained": 0,
    }


def test_immutable_create_only_and_read_only_audit(tmp_path):
    report = _minimal_record()
    out = tmp_path / "study"
    study.create_only(report, output=out, revision="a" * 40)
    with pytest.raises(FileExistsError):
        study.create_only(report, output=out, revision="a" * 40)
    audited = study.audit(out)
    assert audited["e0_certified"] == 3
    assert audited["unknown_e0"] == 1
    assert audited["rejected"] == 2
    assert len(audited["saved_receipt_sha256"]) == 64
    assert audited["original_compiler_or_vendor_calls_during_audit"] == 0


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(e0_certified=4),
    lambda data: data["rows"][0]["action_replay_statuses"].reverse(),
    lambda data: data["rows"][1]["graph_signature"].update(digest="wrong"),
    lambda data: data.update(native_vitis_calls=1),
    lambda data: data.update(finite_candidate_denominator=5),
    lambda data: data.update(source_revision="not-a-sha"),
])
def test_read_only_audit_rejects_tampered_saved_denominators(tmp_path, mutation):
    root = tmp_path / "original"
    study.create_only(_minimal_record(), output=root, revision="b" * 40)
    file = root / "r3d_contract_gate.json"
    record = json.loads(file.read_text())
    mutation(record)
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="R3d:"):
        study.audit(root)


def test_original_revision_and_output_checks_fail_closed(tmp_path):
    record = _minimal_record()
    with pytest.raises(ValueError, match="revision"):
        study.create_only(record, output=tmp_path / "bad", revision="not-git-sha")
    assert not (tmp_path / "bad").exists()
    out = tmp_path / "original"
    out.mkdir()
    with pytest.raises(FileExistsError):
        study.create_only(record, output=out, revision="d" * 40)
