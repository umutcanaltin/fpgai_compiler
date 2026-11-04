"""Actual original Conv HLS source LOAD events and Phi ownership attestation.

The original emitted C++ read address is bound to the trace read address.
No claim to prove complete scalar graph source ordering or HLS RAM ports.
"""
from dataclasses import replace
from hashlib import sha256
import re,shutil
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission,emit_hls
from fpgai_research.experiments.cst_conv_boundary_fixtures import conv_diamond
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.cpp_conv_trace import trace_conv_cpp


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
@pytest.mark.parametrize("geometry",[(4,4,3,1,1),(5,6,3,1,2),(5,6,3,0,1)])
def test_true_emitted_conv_load_indices_and_mapping_are_replayed(geometry):
    h,w,kh,pad,stride=geometry
    model=conv_diamond(height=h,width=w,kernel=kh,pad=pad,stride=stride)
    default=ArchitectureRequest()
    arch=resolve(model,default,{
        "producer":replace(default,pes=2,simd=2,serialization=2),
        "reader1":replace(default,pes=2,simd=2),
    })
    source=emit_hls(lower(arch),trace_conv_access=True)
    receipt=trace_conv_cpp(arch,source)
    assert receipt["status"]=="ACTUAL_CONV_KERNEL_LOADS_VERIFIED_E1"
    assert receipt["source_read_count"]>0
    assert receipt["native_vitis_calls"]==0


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_changed_real_source_index_is_caught_even_if_still_inbounds():
    model=conv_diamond(height=4,width=4,kernel=3,pad=1,stride=1)
    arch=resolve(model,ArchitectureRequest())
    source=emit_hls(lower(arch),trace_conv_access=True)
    before=re.search(r"const int __cst_source_index=[^;]+;",source.cpp)
    assert before is not None and before.group(0).endswith("+ix;")
    old=before.group(0)
    modified=old.replace("+ix;","+((ix+1)%4);")
    altered=source.cpp.replace(old,modified)
    assert altered!=source.cpp
    mutated=Emission(altered,{**source.manifest,
                              "cpp_sha256":sha256(altered.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="ACTUAL_CONV_SOURCE_READ_MISMATCH"):
        trace_conv_cpp(arch,mutated)
    without=emit_hls(lower(arch))
    assert without.manifest.get("conv_source_read_trace_opt_in") is None
    with pytest.raises(ValueError,match="opt-in"):
        trace_conv_cpp(arch,without)
