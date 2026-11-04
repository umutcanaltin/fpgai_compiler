"""Numerical opt-in gate prevents routing E3 cases that failed CSim.

A synthetic runner verifies policy only; no Vitis or Vivado tools execute here.
Existing E3 synthesis remains valid when CSim numerical acceptance fails.
"""
import json
from hashlib import sha256

import numpy as np
import pytest

from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest
from fpgai_research.experiments.campaign import prepare_campaign
from fpgai_research.experiments.execution import run_campaign
from fpgai_research.experiments import execution, suite_execution
from fpgai_research.toolchains.workflow import main


def _prepared(tmp_path):
    m=FunctionalModel((
        Tensor("x",(1,2)),
        Tensor("w",(2,1),data=(0.25,0.75)),
        Tensor("y",(1,1))),
        (Operator("mm","MatMul",("x","w"),("y",)),),
        ("x",),("y",))
    folder=tmp_path/"campaign"
    prepare_campaign(model=m,runtime_inputs=np.array([[1.,2.]],dtype=np.float32),
                     directory=folder,operator="mm",default=ArchitectureRequest(),
                     variants={})
    return folder


def _mock_e3(case, passed):
    return {
        "source_sha256":sha256((case/"model.cpp").read_bytes()).hexdigest(),
        "evidence":"E3-HLS-synthesis-report",
        "csim_passed":passed,
        "vivado_metrics":None,"warnings":[]
    }


@pytest.mark.parametrize("accepted", (None,False))
def test_required_numerical_gate_blocks_e3_with_failed_or_missing_csim(
        tmp_path,monkeypatch,accepted):
    folder=_prepared(tmp_path)
    monkeypatch.setattr(execution,"summarize_case",lambda case:
                        _mock_e3(__import__("pathlib").Path(case),accepted))
    def forbidden(*args,**kwargs):
        raise AssertionError("Vivado must never be invoked for rejected CSim")
    monkeypatch.setattr(execution,"run_vivado",forbidden)
    report=run_campaign(folder,stages=("vivado",),execute=True,
                        require_csim_pass=True,first=0,limit=1)
    status=report["cases"][0]["stages"]["vivado"]
    assert status["status"]=="BLOCKED"
    assert "C simulation" in status["reason"]
    assert report["require_csim_pass"] is True
    assert report["cases"][0]["initial_evidence"]=="E3-HLS-synthesis-report"
    from fpgai_research.experiments.execution_history import verify_execution_history
    assert verify_execution_history(folder)["receipt_count"]==1


def test_gate_allows_a_source_attested_numerically_passing_e3(tmp_path,monkeypatch):
    folder=_prepared(tmp_path)
    monkeypatch.setattr(execution,"summarize_case",lambda case:
                        _mock_e3(__import__("pathlib").Path(case),True))
    calls=[]
    def stub_tool(*args,**kwargs):
        calls.append((args,kwargs))
        return {"completed_successfully":True,"schema":"fpgai-research/vendor-invocation/v1","returncode":0}
    monkeypatch.setattr(execution,"run_vivado",stub_tool)
    report=run_campaign(folder,stages=("vivado",),execute=True,
                        require_csim_pass=True,first=0,limit=1)
    assert report["cases"][0]["stages"]["vivado"]["status"]=="COMPLETED"
    assert len(calls)==1
    assert report["require_csim_pass"] is True


def test_default_gate_remains_backward_compatible(tmp_path,monkeypatch):
    folder=_prepared(tmp_path)
    monkeypatch.setattr(execution,"summarize_case",lambda case:
                        _mock_e3(__import__("pathlib").Path(case),False))
    called=[]
    def stub_tool(*args,**kwargs):
        called.append(1)
        return {"completed_successfully":True,"returncode":0}
    monkeypatch.setattr(execution,"run_vivado",stub_tool)
    report=run_campaign(folder,stages=("vivado",),execute=True)
    assert report["require_csim_pass"] is False
    assert report["cases"][0]["stages"]["vivado"]["status"]=="COMPLETED"
    assert len(called)==1


def test_required_gate_rejected_for_vitis_only_or_non_boolean(tmp_path):
    folder=_prepared(tmp_path)
    with pytest.raises(ValueError,match="requires a Vivado"):
        run_campaign(folder,stages=("vitis",),require_csim_pass=True)
    with pytest.raises(ValueError,match="must be a boolean"):
        run_campaign(folder,stages=("vivado",),require_csim_pass=1)


def test_dry_run_required_gate_exposes_block_without_invoking_tools(
        tmp_path,monkeypatch):
    folder=_prepared(tmp_path)
    monkeypatch.setattr(execution,"summarize_case",lambda case:
                        _mock_e3(__import__("pathlib").Path(case),False))
    monkeypatch.setattr(execution,"run_vivado",lambda *a,**kw:
                        (_ for _ in ()).throw(AssertionError("Tool called in dry run")))
    report=run_campaign(folder,stages=("vivado",),require_csim_pass=True)
    assert report["tools_executed"] is False
    assert report["cases"][0]["stages"]["vivado"]["status"]=="BLOCKED"
    assert not (folder/"last_campaign_execution.json").exists()


def test_suite_gate_propagates_to_campaigns(tmp_path,monkeypatch):
    root=tmp_path/"suite"
    root.mkdir()
    (root/"suite_results.json").write_text(json.dumps({"studies":[{"id":"mlp_w8i4"},{"id":"cnn_w8i4"}]}))
    monkeypatch.setattr(suite_execution,"audit_suite",lambda *args,**kwargs: {"all_models_source_ready":True})
    args=[]
    def mock_campaign(path,**kwargs):
        args.append((path,kwargs["require_csim_pass"]))
        return {"cases":[{"stages":{"vivado":{"status":"PLANNED-NOT-EXECUTED"}}}]}
    monkeypatch.setattr(suite_execution,"run_campaign",mock_campaign)
    result=suite_execution.run_full_suite(root,stages=("vivado",),
                                          require_csim_pass=True,execute=False)
    assert result["require_csim_pass"] is True
    assert [part[1] for part in args]==[True,True]
    with pytest.raises(ValueError,match="requires Vivado"):
        suite_execution.run_full_suite(root,stages=("vitis",),require_csim_pass=True)


def test_cli_passes_opt_in_gate_to_campaign_and_suite(monkeypatch,capsys):
    observed=[]
    def mock_campaign(*args,**kwargs):
        observed.append(("campaign",kwargs["require_csim_pass"]))
        return {"cases":[{"stages":{"vivado":{"status":"BLOCKED"}}}]}
    def mock_suite(*args,**kwargs):
        observed.append(("suite",kwargs["require_csim_pass"]))
        return {"models":[{"execution":{"cases":[{"stages":{"vivado":{"status":"BLOCKED"}}}]}}]}
    monkeypatch.setattr(execution,"run_campaign",mock_campaign)
    monkeypatch.setattr(suite_execution,"run_full_suite",mock_suite)
    assert main(["run-campaign","--directory","fake","--stage","vivado",
                 "--require-csim-pass","--execute"])==1
    capsys.readouterr()
    assert main(["run-suite","--directory","fake","--stage","vivado",
                 "--require-csim-pass","--execute"])==1
    assert observed==[("campaign",True),("suite",True)]
