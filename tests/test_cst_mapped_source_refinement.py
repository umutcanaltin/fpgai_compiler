"""E0 Phi mapped event proof against independent executed E1 compiler output."""
import shutil
import pytest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.mapped_source_refinement import qualify_mapped_source

@pytest.mark.parametrize("rows,block",[(2,1),(3,2),(5,2)])
def test_actual_cpp_matches_independently_qualified_logical_lifetimes(rows,block):
    if shutil.which("g++") is None:
        pytest.skip("Real GCC source compiler needed")
    graph=make_chain(rows,(3,5,4,2))
    arch=resolve(graph,ArchitectureRequest())
    e=qualify_mapped_source(arch,row_block=block)
    assert e["status"]=="EXECUTED_SOURCE_REFINEMENT_VERIFIED"
    assert e["logical_value_count"]==rows*(5+4)
    assert e["actual_cpp_last_reader_releases"]==rows*(5+4)
    assert e["actual_cpp_iteration_count"]>0
    assert e["original_vitis_calls"]==0
    assert len(e["receipt_sha256"])==64

def test_bounded_trace_declares_unverified_instead_of_pass():
    graph=make_chain(3,(3,5,4,2))
    arch=resolve(graph,ArchitectureRequest())
    with pytest.raises(ValueError,match="bound"):
        qualify_mapped_source(arch,row_block=2,max_events=2)
