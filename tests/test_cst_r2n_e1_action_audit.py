"""R2n independent saved-receipt consistency, not a GCC replay."""
from __future__ import annotations

from copy import deepcopy
import json
from hashlib import sha256

import numpy as np
import pytest

from fpgai_research.experiments import cst_r2n_e1_action_audit as sut


def _fixture(tmp_path):
    root = tmp_path / "r2n"
    root.mkdir()
    src, cpp = "b"*64, "c"*64
    input_sha = sha256(np.linspace(
        -.125, .125, 9, dtype=np.float32).tobytes()).hexdigest()
    normal = {
        "action_index": 0,
        "action": {"scope":"operator", "target":"mm0", "field":"pes", "value":2},
        "source_admissible": True,
        "e0_mapped_certified": True,
        "e1_host_source_qualified": True,
        "status": "E1_GCC_SOURCE_QUALIFIED",
        "reason": None,
        "state_sha256": src,
        "cpp_sha256": cpp,
        "qualification_sha256": "e"*64,
        "host_input_sha256": input_sha,
        "host_max_absolute_error": 1e-7,
        "host_rmse": 1e-8,
        "source_trace_sha256": "f"*64,
        "last_reader_releases_verified": 9,
    }
    reject = {
        **normal,
        "action_index": 1,
        "action": {"scope":"operator", "target":"mm0", "field":"pes", "value":0},
        "source_admissible": False,
        "e0_mapped_certified": False,
        "e1_host_source_qualified": False,
        "status": "REJECTED",
        "reason": "Invalid PE",
        "state_sha256": None,
        "cpp_sha256": None,
        "qualification_sha256": None,
        "host_input_sha256": None,
        "host_max_absolute_error": None,
        "host_rmse": None,
        "source_trace_sha256": None,
        "last_reader_releases_verified": None,
    }
    unknown = {
        **reject,
        "action_index": 2,
        "action": {"scope":"exchange", "target":"model",
                   "field":"exchange_policy", "value":"whole_tensor"},
        "source_admissible": True,
        "status": "UNKNOWN_E0_CERTIFICATE",
        "reason": "No mapped value proof",
        "state_sha256": src,
        "cpp_sha256": cpp,
    }
    result = {
        "schema":"fpgai-research/cst-R2n-original-GCC-certified-C3-decisions/v1",
        "source_revision":"a"*40,
        "graphs":1, "graph_depths":[2],
        "all_candidate_actions":3,
        "status_denominators":{
            "E1_GCC_SOURCE_QUALIFIED":1, "REJECTED":1,
            "UNKNOWN_E0_CERTIFICATE":1},
        "e1_host_executions_attempted":1,
        "e1_verified_actions":1,
        "source_only_uncertified_actions":1,
        "rejected_actions":1, "failed_e1_actions":0,
        "native_vitis_calls":0, "new_vivado_calls":0,
        "rl_policies_trained":0,
    }
    detail = [{
        "graph_depth":2,
        "roles":["mapped_Phi_edit","invalid_Phi_edit",
                 "source_only_unverified_exchange"],
        "fixed_input_sha256":input_sha,
        "original_state_sha256":"1"*64,
        "e1_attempts":1,
        "native_vitis_calls":0, "rl_policies_trained":0,
        "actions":[normal, reject, unknown],
    }]
    (root/"summary.json").write_text(json.dumps(result))
    (root/"case_evidence.json").write_text(json.dumps(detail))
    return root, result, detail


def _rewrite(root, summary, detail):
    (root/"summary.json").write_text(json.dumps(summary))
    (root/"case_evidence.json").write_text(json.dumps(detail))


def test_readonly_audit_preserves_entire_denominator(tmp_path):
    root, data, detail = _fixture(tmp_path)
    result = sut.audit(root)
    assert result["original_graphs"] == 1
    assert result["original_action_cases"] == 3
    assert result["source_e1_qualifications"] == 1
    assert result["source_e1_qualification_attempts"] == 1
    assert result["native_vendor_calls_during_audit"] == 0
    assert data == json.loads((root/"summary.json").read_text())
    assert detail == json.loads((root/"case_evidence.json").read_text())


def test_missing_original_source_trace_cannot_promote_e1(tmp_path):
    root, data, detail = _fixture(tmp_path)
    detail[0]["actions"][0]["source_trace_sha256"] = None
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="original GCC evidence"):
        sut.audit(root)


def test_unknown_source_admission_cannot_promote_to_certified(tmp_path):
    root, data, detail = _fixture(tmp_path)
    detail[0]["actions"][2]["e0_mapped_certified"] = True
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="unknown E0 action"):
        sut.audit(root)


def test_bad_original_denominators_and_status_order_are_rejected(tmp_path):
    root, data, detail = _fixture(tmp_path)
    data["status_denominators"]["REJECTED"] = 2
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="denominator"):
        sut.audit(root)
    data["status_denominators"]["REJECTED"] = 1
    detail[0]["actions"] = detail[0]["actions"][:2]
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="incomplete graph/action"):
        sut.audit(root)


def test_corrupt_source_pin_and_original_action_treatment_rejected(tmp_path):
    root, data, detail = _fixture(tmp_path)
    data["source_revision"] = "not-full-commit"
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="source pin"):
        sut.audit(root)
    data["source_revision"] = "a"*40
    detail[0]["actions"][1]["action"]["value"] = 4
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="original Phi treatment"):
        sut.audit(root)


def test_symlinked_evidence_and_output_overwrite_rejected(tmp_path):
    root, data, detail = _fixture(tmp_path)
    alias = tmp_path/"alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        sut.audit(alias)
    evidence = root/"case_evidence.json"
    evidence.unlink()
    evidence.symlink_to(root/"summary.json")
    with pytest.raises(ValueError, match="missing/symlinked"):
        sut.audit(root)



def test_paper_input_digest_is_independently_recomputed(tmp_path):
    root, data, detail = _fixture(tmp_path)
    detail[0]["fixed_input_sha256"] = "d" * 64
    detail[0]["actions"][0]["host_input_sha256"] = "d" * 64
    _rewrite(root, data, detail)
    with pytest.raises(ValueError, match="deterministic GCC input SHA"):
        sut.audit(root)
