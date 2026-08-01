from fpgai.operators import OnnxBinding, OperatorCapabilities, OperatorContract, TensorPortContract
from fpgai.operators.external import ExternalOperatorDefinition, ImportedOperator, OnnxBindingRegistry, RegisteredOnnxBinding

def _item(package, lo, hi):
    c=OperatorContract(f"{package}.operator.test","Test",1,"test",(TensorPortContract("x"),),(TensorPortContract("y"),),onnx_bindings=(OnnxBinding("community","Test",lo,hi),),capabilities=OperatorCapabilities(inference=True))
    d=ExternalOperatorDefinition(1,c,lambda x:ImportedOperator("Test",x.inputs,x.outputs,x.attributes))
    return RegisteredOnnxBinding(package,"1.0.0","sha256:x",c.onnx_bindings[0],d)

def test_binding_lookup_and_overlap_detection():
    r=OnnxBindingRegistry(); assert not r.register(_item("community.a",1,3)); assert r.resolve("community","Test",2)
    assert r.register(_item("community.b",3,5))[0].code=="OPLOAD009"
