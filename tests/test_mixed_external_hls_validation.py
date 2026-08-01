from __future__ import annotations
from pathlib import Path
import numpy as np

from fpgai.ir.graph import Graph
from fpgai.operators.external.external_api import ReferenceExecutionResult
from fpgai.validation.mixed_external_hls import execute_mixed_graph_reference, prepare_mixed_external_validation


class _Context:
    def reference_for(self, operator_id):
        assert operator_id == "community.operator.scale_bias"
        def _run(ctx):
            x = ctx.inputs[0]
            return ReferenceExecutionResult((x * float(ctx.attributes["scale"]) + float(ctx.attributes["bias"]),))
        return _run


def _graph():
    g = Graph("mixed")
    g.inputs=["x"]; g.outputs=["z"]
    for name in ("x","r","s","z"): g.add_tensor(name,(1,4),"float32")
    g.add_op("Relu",["x"],["r"],"relu_0")
    g.add_op("ScaleBias",["r"],["s"],"scale_bias_0",{
        "scale":2.0,"bias":1.0,
        "_fpgai_external_operator":{"operator_id":"community.operator.scale_bias"},
    })
    g.add_op("Sigmoid",["s"],["z"],"sigmoid_0")
    return g


def test_reference_executes_builtin_external_builtin_sequence():
    actual=execute_mixed_graph_reference(_graph(),_Context(),np.array([-2,-.5,.5,2],dtype=np.float32))
    expected=1.0/(1.0+np.exp(-(np.maximum(np.array([-2,-.5,.5,2],dtype=np.float32),0)*2+1)))
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def test_prepare_writes_reproducible_binary_and_report(tmp_path: Path):
    result=prepare_mixed_external_validation(graph=_graph(),external_context=_Context(),out_dir=tmp_path,config={"input_values":[-2,-.5,.5,2]})
    assert result.input_bin.is_file()
    assert result.expected_bin.is_file()
    assert result.report_path.is_file()
    assert np.fromfile(result.input_bin,dtype=np.float32).tolist()==[-2.0,-0.5,0.5,2.0]


def test_portable_host_cpp_executes_package_source(tmp_path: Path):
    import shutil
    if not (shutil.which("g++") or shutil.which("c++")):
        import pytest; pytest.skip("C++ compiler unavailable")
    from types import SimpleNamespace
    from fpgai.validation.mixed_external_hls import run_portable_host_cpp_validation
    hls_dir=tmp_path/"hls"
    src_dir=hls_dir/"src/external/pkg"; inc_dir=hls_dir/"include/external/pkg"
    src_dir.mkdir(parents=True); inc_dir.mkdir(parents=True)
    (inc_dir/"scale_bias.hpp").write_text("void scale_bias_hls(const float*, float*, int, float, float);\n")
    (src_dir/"scale_bias.cpp").write_text('#include "scale_bias.hpp"\nvoid scale_bias_hls(const float* x,float* y,int n,float s,float b){for(int i=0;i<n;++i)y[i]=x[i]*s+b;}\n')
    artifacts=prepare_mixed_external_validation(graph=_graph(),external_context=_Context(),out_dir=tmp_path,config={"input_values":[-2,-.5,.5,2]})
    contract=SimpleNamespace(top="scale_bias_hls",metadata={"integration":{"hls":{"attributes":[{"name":"scale","cpp_type":"float","default":1.0},{"name":"bias","cpp_type":"float","default":0.0}]}}})
    binding=SimpleNamespace(contract=contract,attributes={"scale":2.0,"bias":1.0})
    plan=SimpleNamespace(bindings=(binding,),binding_for_node=lambda name: binding if name=="scale_bias_0" else None)
    result=run_portable_host_cpp_validation(graph=_graph(),composition_plan=plan,artifacts=artifacts,hls_dir=hls_dir)
    assert result["status"]=="passed"
