"""Check mathematical semantics at each intermediate Model IR tensor, not final output alone."""
import shutil
import numpy as np
import pytest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.tensor_trace import check_tensor_trace
from fpgai_research.experiments.branch_ir_study import make_branched_conv_model

pytestmark = pytest.mark.skipif(shutil.which('g++') is None, reason='Host C++ compiler unavailable')


def test_all_intermediate_tensors_of_branched_graph_match_numpy():
    model, x = make_branched_conv_model()
    base = check_tensor_trace(lower(resolve(model,ArchitectureRequest())), x)
    specialized = ArchitectureRequest(pes=2,parallelism=2,simd=2,serialization=2,
                                      tile=(1,2,2,3,1,1,1))
    variant = check_tensor_trace(lower(resolve(model,ArchitectureRequest(),
                                                {'conv_a':specialized})),x)
    assert base['accepted'] and variant['accepted']
    assert base['operator_count'] == variant['operator_count'] == 5
    assert {p['operator'] for p in base['operators']} == {'conv_a','conv_b','join','relu','pool'}
    assert base['total_values'] == variant['total_values']
    assert base['cpp_sha256'] != variant['cpp_sha256']
    assert all(op['max_absolute_error'] < 1e-6 for op in base['operators'])
    assert all(op['max_absolute_error'] < 1e-6 for op in variant['operators'])
    assert not base['native_vitis_executed']


def test_repeated_canonical_trace_is_deterministic():
    model,x=make_branched_conv_model()
    lowered=lower(resolve(model,ArchitectureRequest()))
    a=check_tensor_trace(lowered,x)
    b=check_tensor_trace(lowered,x)
    assert a['raw_trace_sha256']==b['raw_trace_sha256']


def test_trace_budget_and_shape_are_checked_before_compilation():
    model,x=make_branched_conv_model()
    lowered=lower(resolve(model,ArchitectureRequest()))
    with pytest.raises(ValueError,match='bound'):
        check_tensor_trace(lowered,x,max_values=2)
    with pytest.raises(ValueError,match='exactly shaped'):
        check_tensor_trace(lowered,np.ones((1,3),np.float32))
