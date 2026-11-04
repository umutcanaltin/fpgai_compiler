"""Real ONNX checker/shape inference (optional integration dependency).

Exercises static Constant->Reshape->Gemm(transB)->broadcast Add->Relu,
then validates independent NumPy and generated GCC host C++ results.
"""
import shutil
import numpy as np
import pytest

def test_native_onnx_export_semantics(tmp_path):
    onnx=pytest.importorskip("onnx")
    from onnx import helper,numpy_helper,TensorProto
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.validation.reference import evaluate
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.model_ir.lowering import lower
    from fpgai_research.codegen.hls.emitter import emit_hls
    from fpgai_research.validation.host import validate_host
    w=np.arange(24,dtype=np.float32).reshape(4,6)/32
    bias=np.array([.1,-.2,.3,-.4],dtype=np.float32)
    shape=numpy_helper.from_array(np.array([0,-1],dtype=np.int64))
    nodes=[
      helper.make_node("Constant",[],["shape"],name="shape_constant",value=shape),
      helper.make_node("Reshape",["x","shape"],["flat"],name="reshape"),
      helper.make_node("Gemm",["flat","weight"],["mm"],name="linear",transB=1),
      helper.make_node("Add",["mm","bias"],["added"],name="bias_add"),
      helper.make_node("Relu",["added"],["y"],name="relu"),
    ]
    graph=helper.make_graph(nodes,"research-static-model",
      [helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,2,3])],
      [helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,4])],
      initializer=[numpy_helper.from_array(w,"weight"),numpy_helper.from_array(bias,"bias")])
    proto=helper.make_model(graph,opset_imports=[helper.make_operatorsetid("",13)],ir_version=8)
    file=tmp_path/"graph.onnx"
    onnx.save(proto,str(file))
    onnx.checker.check_model(str(file))
    model=load_onnx(str(file))
    assert [op.kind for op in model.operators]==["Reshape","Gemm","Add","Relu"]
    x=np.arange(6,dtype=np.float32).reshape(1,2,3)/16
    oracle=evaluate(model,{"x":x})["y"]
    if shutil.which("g++"):
        emitted=emit_hls(lower(resolve(model,ArchitectureRequest())))
        result=validate_host(model,emitted,x)
        assert result["accepted"],result
    try:
        import onnxruntime as ort
    except ImportError:
        return
    independent=ort.InferenceSession(str(file),providers=["CPUExecutionProvider"]).run(None,{"x":x})[0]
    np.testing.assert_allclose(oracle,independent,rtol=2e-6,atol=2e-6)


def test_real_onnx_static_squeeze_unsqueeze_axes(tmp_path):
    import shutil
    onnx=pytest.importorskip("onnx")
    from onnx import TensorProto,helper,numpy_helper
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.validation.reference import evaluate
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.model_ir.lowering import lower
    from fpgai_research.codegen.hls.emitter import emit_hls
    from fpgai_research.validation.host import validate_host
    axes=np.array([0,2],dtype=np.int64)
    nodes=[
      helper.make_node("Constant",[],["sq_axes"],value=numpy_helper.from_array(axes)),
      helper.make_node("Squeeze",["x","sq_axes"],["sq"]),
      helper.make_node("Unsqueeze",["sq","us_axes"],["us"]),
      helper.make_node("Identity",["us"],["y"]),
    ]
    graph=helper.make_graph(nodes,"static-axes",
       [helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,2,1,3])],
       [helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,2,1,3])],
       initializer=[numpy_helper.from_array(axes,name="us_axes")])
    proto=helper.make_model(graph,opset_imports=[helper.make_operatorsetid("",13)],ir_version=8)
    path=tmp_path/"axes.onnx"
    onnx.save(proto,str(path))
    onnx.checker.check_model(str(path))
    model=load_onnx(str(path))
    assert [op.kind for op in model.operators]==["Squeeze","Unsqueeze","Identity"]
    x=np.arange(6,dtype=np.float32).reshape(1,2,1,3)
    np.testing.assert_array_equal(evaluate(model,{"x":x})["y"],x)
    if shutil.which("g++"):
        result=validate_host(model,emit_hls(lower(resolve(model,ArchitectureRequest()))),x)
        assert result["accepted"],result
    try:
        import onnxruntime as ort
    except ImportError:
        return
    output=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"]).run(None,{"x":x})[0]
    np.testing.assert_array_equal(output,x)
