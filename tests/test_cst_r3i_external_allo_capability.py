"""R3i fail-closed external Allo protocol: no implicit external invocation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fpgai_research.experiments import cst_r3i_external_allo_capability as sut


def test_read_only_protocol_does_not_execute_competitor():
    p=sut.protocol()
    assert p["schema"]==sut.SCHEMA
    assert p["external_source_revision"]==sut.PINNED_ALLO
    assert p["result_status"]=="NOT_RUN"
    assert p["external_artifact_executed"] is False
    assert p["original_vitis_calls"]==0
    assert p["trained_rl_agents"]==0
    assert len(p["capability_questions"])==3


def test_original_numpy_reference_demonstrates_real_negative_perturbation():
    p=sut.reference_outputs()
    x,wp,wa,wb=sut.reference_inputs()
    expected=(x@wp)@wa + (x@wp)@wb
    broken=(x@wp)@wa
    assert np.allclose(np.array(p["reference_good"]),expected,atol=1e-6)
    assert np.allclose(np.array(p["reference_early_overwrite"]),broken,atol=1e-6)
    assert not np.allclose(expected,broken,atol=1e-6)
    assert p["dtype"]=="float32"
    assert len(p["input_sha256"])==64


def test_external_checkout_must_be_pinned_before_import(monkeypatch,tmp_path):
    # Does not import/execute Allo on the normal research pytest path.
    monkeypatch.setattr(sut,"_git_revision",
                        lambda *_args: (_ for _ in ()).throw(
                            ValueError("R3i Allo revision mismatch")))
    with pytest.raises(ValueError,match="revision mismatch"):
        sut.execute_allo(tmp_path)


def test_new_artifact_report_is_create_only_and_no_fabricated_result(tmp_path):
    target=tmp_path/"original"
    output=sut.protocol()
    sut.create_only(output,output=target)
    assert json.loads((target/"r3i_pinned_allo_capability.json").read_text())[
        "result_status"]=="NOT_RUN"
    with pytest.raises(FileExistsError,match="create-only"):
        sut.create_only(output,output=target)
    with pytest.raises(ValueError,match="schema"):
        sut.create_only({"schema":"invented"},output=tmp_path/"broken")


def test_no_unconfirmed_external_execution(capsys):
    with pytest.raises(SystemExit):
        sut.main(["--allo-checkout","/some/checkout","--output","some_output"])
    sut.main(["--protocol"])
    printed=json.loads(capsys.readouterr().out)
    assert printed["external_artifact_executed"] is False
