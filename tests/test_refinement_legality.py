"""Adversarial snapshots cannot be rescued by stage-internal consistency."""
from dataclasses import replace
import pytest
from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel, ArchitectureRequest, ArchitecturalModel, LoweredModel
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import verify_refinement

def fixture():
    f=FunctionalModel((Tensor('x',(1,4)),Tensor('weights',(4,3),data=(.1,)*12),Tensor('y',(1,3))),
      (Operator('op','MatMul',('x','weights'),('y',)),),('x',),('y',))
    return lower(resolve(f,ArchitectureRequest()))

def test_reject_same_wrong_iteration_geometry_across_architecture_and_lowering():
    original=fixture()
    p=replace(original.architectural.points[0],iteration_shape=(1,4,4))
    op=replace(original.operators[0],iteration_shape=(1,4,4),tile_shape=(1,4,4))
    forged=LoweredModel(ArchitecturalModel(original.architectural.functional,(p,)),(op,))
    with pytest.raises(ValueError,match='MODELIR-SHAPE'):
        verify_refinement(forged)

def test_reject_illegal_non_compute_tile_even_with_matching_stage_mechanisms():
    f=FunctionalModel((Tensor('x',(1,4)),Tensor('y',(1,4))),
      (Operator('relu','Relu',('x',),('y',)),),('x',),('y',))
    original=lower(resolve(f,ArchitectureRequest()))
    illegal=replace(ArchitectureRequest(),tile=(1,4))
    point=replace(original.architectural.points[0],requested=illegal,effective=illegal)
    op=replace(original.operators[0],architecture=illegal,tile_shape=(1,4))
    forged=LoweredModel(ArchitecturalModel(f,(point,)),(op,))
    with pytest.raises(ValueError,match='CAPABILITY'):
        verify_refinement(forged)

def test_reject_duplicate_provenance_identity():
    original=fixture()
    point=replace(original.architectural.points[0],decision_sources=(('pes','model'),('pes','model')))
    op=replace(original.operators[0],decision_sources=point.decision_sources)
    forged=LoweredModel(ArchitecturalModel(original.architectural.functional,(point,)),(op,))
    with pytest.raises(ValueError,match='MODELIR-PROVENANCE'):
        verify_refinement(forged)

def test_unmodified_artifact_remains_legal():
    assert verify_refinement(fixture())['stage_invariants_verified']
