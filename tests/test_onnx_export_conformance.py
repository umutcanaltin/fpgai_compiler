"""Isolated E0 exported-ONNX numerical agreement; no Vitis/Vivado or Torch needed."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from fpgai_research.experiments.onnx_export_conformance import validate_exported_model


def _fixture(monkeypatch,tmp_path,*,mode="valid"):
    onnx=tmp_path/"model.onnx"
    onnx.write_bytes(b"ONNX isolated conformance fixture (not an actual graph)")
    values=np.array([[[1., 2.]], [[3., 4.]]],dtype=np.float32)
    expected=values.reshape(2,2).copy()
    class FakeSession:
        def __init__(self,*args,**kwargs):
            self.args=args
        def get_inputs(self):
            return [SimpleNamespace(name="x",shape=[1,1,2])]
        def get_outputs(self):
            return [SimpleNamespace(name="y",shape=[1,2])]
        def run(self,unused,feeds):
            value=feeds["x"].copy().reshape(1,2)
            if mode=="wrong":
                value+=np.float32(0.1)
            elif mode=="nonfinite":
                value.fill(np.inf)
            elif mode=="wrong-shape":
                value=value[:,:1]
            elif mode=="wrong-dtype":
                value=value.astype(np.float64)
            elif mode=="near-tie":
                value=np.array([[.500001,.5]],dtype=np.float32)
            return [value]
    monkeypatch.setitem(sys.modules,"onnxruntime",
                        SimpleNamespace(__version__="test-only",InferenceSession=FakeSession))
    return onnx,values,expected


def test_onnxruntime_reference_agreement_is_explicit_e0(monkeypatch,tmp_path):
    model,x,expected=_fixture(monkeypatch,tmp_path)
    result=validate_exported_model(model,x,expected,check_model_ir=False)
    assert result["evaluated_samples"]==2
    assert result["onnxruntime_max_abs_error"]==0.0
    assert result["class_predictions_matched"] is True
    assert not result["native_vitis_executed"] and not result["vivado_executed"]
    assert not result["model_ir_compared"]
    assert result["evidence"]=="E0-PyTorch-ONNXRuntime-only"


@pytest.mark.parametrize("mode,message",[
    ("wrong","numerically"),
    ("nonfinite","nonfinite"),
    ("wrong-shape","invalid"),
    ("wrong-dtype","invalid"),
])
def test_malformed_or_wrong_onnx_outputs_cannot_be_accepted(monkeypatch,tmp_path,mode,message):
    model,x,expected=_fixture(monkeypatch,tmp_path,mode=mode)
    with pytest.raises(ValueError,match=message):
        validate_exported_model(model,x,expected,check_model_ir=False)


def test_model_ir_boundary_and_values_checked(monkeypatch,tmp_path):
    import fpgai_research.frontend.onnx as frontend
    import fpgai_research.validation.reference as numeric
    model,x,expected=_fixture(monkeypatch,tmp_path)
    ir=SimpleNamespace(inputs=("x",),outputs=("y",),
                       tensor=lambda name:SimpleNamespace(shape=(1,1,2) if name=="x" else (1,2)))
    monkeypatch.setattr(frontend,"load_onnx",lambda *_:ir)
    monkeypatch.setattr(numeric,"evaluate",lambda *_a,**_kw:{
        "y":_a[1]["x"].reshape(1,2)})
    result=validate_exported_model(model,x,expected,check_model_ir=True)
    assert result["model_ir_compared"] is True
    assert result["model_ir_max_abs_error"]==0.0
    assert result["evidence"]=="E0-PyTorch-ONNXRuntime-ModelIR"


def test_disagreeing_compiler_ir_rejected(monkeypatch,tmp_path):
    import fpgai_research.frontend.onnx as frontend
    import fpgai_research.validation.reference as numeric
    model,x,expected=_fixture(monkeypatch,tmp_path)
    ir=SimpleNamespace(inputs=("x",),outputs=("y",),
                       tensor=lambda name:SimpleNamespace(shape=(1,1,2) if name=="x" else (1,2)))
    monkeypatch.setattr(frontend,"load_onnx",lambda *_:ir)
    monkeypatch.setattr(numeric,"evaluate",lambda *_a,**_kw:{
        "y":np.ones((1,2),dtype=np.float32)*100})
    with pytest.raises(ValueError,match="Compiler Model IR output differs"):
        validate_exported_model(model,x,expected,check_model_ir=True)


def test_implicit_dtype_conversion_rejected(monkeypatch,tmp_path):
    model,x,expected=_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match="exact float32"):
        validate_exported_model(model,x.astype(np.float64),expected,check_model_ir=False)


@pytest.mark.parametrize("invalid", ["NaN","negative","text"])
def test_invalid_tolerance_policy_rejected(monkeypatch,tmp_path,invalid):
    model,x,expected=_fixture(monkeypatch,tmp_path)
    tol={"NaN":float("nan"),"negative":-1.0,"text":"0.1"}[invalid]
    with pytest.raises(ValueError,match="conformance policy"):
        validate_exported_model(model,x,expected,check_model_ir=False,atol=tol)


def test_near_tie_class_flip_cannot_pass_small_numeric_error(monkeypatch,tmp_path):
    model,x,_=_fixture(monkeypatch,tmp_path,mode="near-tie")
    reference=np.array([[.5,.500001],[.5,.500001]],dtype=np.float32)
    with pytest.raises(ValueError,match="class predictions disagree"):
        validate_exported_model(model,x,reference,check_model_ir=False)
