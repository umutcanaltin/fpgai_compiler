from pathlib import Path
import pytest

def test_external_scale_bias_import(tmp_path: Path):
    onnx=pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from fpgai.frontend.onnx import import_onnx
    from fpgai.registries import RegistryCatalogue, RegistrySource
    from fpgai.operators.external import OperatorLoadRequest, load_operator_packages
    c=RegistryCatalogue(); c.register_package("examples/packages/scale_bias_operator",RegistrySource.PROJECT_LOCAL)
    loaded=load_operator_packages(OperatorLoadRequest(c,("community.scale_bias_operator",),"approved_for_reference")); assert loaded.ok
    node=helper.make_node("ScaleBias",["x"],["y"],domain="community.fpgai",scale=2.0,bias=1.0)
    graph=helper.make_graph([node],"g",[helper.make_tensor_value_info("x",TensorProto.FLOAT,[1,4])],[helper.make_tensor_value_info("y",TensorProto.FLOAT,[1,4])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",13),helper.make_opsetid("community.fpgai",1)])
    path=tmp_path/"m.onnx"; onnx.save(model,path)
    ir=import_onnx(str(path),external_operator_context=loaded.context)
    assert ir.ops[0].op_type=="ScaleBias"
    assert ir.ops[0].attrs["scale"]==2.0
