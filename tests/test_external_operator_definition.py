from fpgai.operators import OnnxBinding, OperatorCapabilities, OperatorContract, TensorPortContract
from fpgai.operators.external import ExternalOperatorDefinition, ImportedOperator

def test_external_definition_requires_api_v1():
    contract=OperatorContract("community.operator.test","Test",1,"test",(TensorPortContract("x"),),(TensorPortContract("y"),),onnx_bindings=(OnnxBinding("community","Test",1),),capabilities=OperatorCapabilities(inference=True))
    definition=ExternalOperatorDefinition(1,contract,lambda c: ImportedOperator("Test",c.inputs,c.outputs,c.attributes))
    assert definition.contract.operator_id=="community.operator.test"
