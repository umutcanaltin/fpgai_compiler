"""Model, region, operator and supported loop semantics with provenance."""
import pytest
from fpgai_research import ArchitectureRequest, FunctionalModel, Operator, Tensor, lower
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.architecture.resolver import ArchitectureError

def graph():
    return FunctionalModel((Tensor("x",(2,3)),
      Tensor("w",(3,4),data=tuple(float(i) for i in range(12))),
      Tensor("z",(2,4)),Tensor("y",(2,4))),(
      Operator("mm","MatMul",("x","w"),("z",)),
      Operator("relu","Relu",("z",),("y",))),("x",),("y",))

def test_per_field_region_operator_loop_priority():
    arch=resolve_scoped(graph(),{"precision_bits":12,"integer_bits":5},
      regions={"stem":{"operators":["mm"],"architecture":{"parallelism":2,"pes":2}}},
      operators={"mm":{"pes":3}},
      loops={"mm.k":{"simd":4,"pipeline_ii":2,"tile_extent":2},
             "mm.n":{"tile_extent":3}})
    p=arch.points[0]
    assert (p.effective.precision_bits,p.effective.parallelism,p.effective.pes,p.effective.simd)==(12,2,3,4)
    assert p.effective.tile==(2,3,2)
    assert p.source_scope=="loop"
    sources=dict(p.decision_sources)
    assert sources["pes"]=="operator:mm"
    assert sources["parallelism"]=="region:stem"
    assert sources["simd"]=="loop:mm.k"
    assert sources["precision_bits"]=="model"
    assert arch.points[1].source_scope=="model"
    emission=emit_hls(lower(arch))
    assert emission.manifest["operators"][0]["decision_sources"]["simd"]=="loop:mm.k"
    assert "for (int lane=0; lane<4" in emission.cpp

def test_invalid_loop_settings_blocked():
    with pytest.raises(ArchitectureError,match="LOOP_CAPABILITY"):
        resolve_scoped(graph(),{},loops={"mm.m":{"pipeline_ii":2}})
    with pytest.raises(ArchitectureError,match="LOOP_SCOPE"):
        resolve_scoped(graph(),{},loops={"mm.missing":{"simd":3}})

def test_ambiguous_region_blocked():
    with pytest.raises(ArchitectureError,match="AMBIGUOUS_REGION"):
        resolve_scoped(graph(),{},regions={
          "one":{"operators":["mm"]},"two":{"operators":["mm"]}})

def test_independent_per_operator_override():
    a=resolve_scoped(graph(),{"precision_bits":16},operators={"mm":{"pes":2}})
    assert a.points[0].effective.pes==2
    assert a.points[1].effective.pes==1
