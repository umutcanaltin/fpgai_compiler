from types import SimpleNamespace
from fpgai.implementations.mixed_backend import build_mixed_backend_plan


def test_mixed_backend_plan_marks_hls_vhdl_boundary():
    graph=SimpleNamespace(ops=[SimpleNamespace(name='relu'),SimpleNamespace(name='custom'),SimpleNamespace(name='add')])
    vhdl=SimpleNamespace(backend='vhdl')
    plan=build_mixed_backend_plan(graph,{'custom':vhdl})
    assert [x['backend'] for x in plan['segments']]==['vitis_hls','vhdl','vitis_hls']
    assert len(plan['bridges'])==2
    assert plan['direct_mixed_rtl_emission_supported'] is False
