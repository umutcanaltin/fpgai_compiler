from pathlib import Path
import numpy as np
from fpgai.registries import RegistryCatalogue, RegistrySource
from fpgai.operators.external import OperatorLoadRequest, ReferenceExecutionContext, load_operator_packages

EXAMPLE=Path("examples/packages/scale_bias_operator")

def test_loader_requires_explicit_trust():
    c=RegistryCatalogue(); assert c.register_package(EXAMPLE,RegistrySource.PROJECT_LOCAL).ok
    result=load_operator_packages(OperatorLoadRequest(c,("community.scale_bias_operator",)))
    assert not result.ok and result.errors[0].code=="OPLOAD002"

def test_loader_loads_approved_reference_operator():
    c=RegistryCatalogue(); assert c.register_package(EXAMPLE,RegistrySource.PROJECT_LOCAL).ok
    result=load_operator_packages(OperatorLoadRequest(c,("community.scale_bias_operator",),"approved_for_reference"))
    assert result.ok
    callback=result.context.reference_for("community.operator.scale_bias")
    output=callback(ReferenceExecutionContext({"scale":2.0,"bias":1.0},(np.array([1,2],dtype=np.float32),))).outputs[0]
    assert output.tolist()==[3.0,5.0]
