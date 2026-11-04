"""Actual generated NCHW intermediate tensor E1 trace and causal mutation gates."""
from dataclasses import replace
from hashlib import sha256
import re
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls,Emission
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.validation.cpp_nchw_intermediate_trace import (
    attest_nchw_intermediates,
)


def setup_source(kind="branch_concat",different=True):
    model=nchw_case(kind,height=4,width=4)
    base=ArchitectureRequest()
    overrides={}
    if different:
        for i,op in enumerate(model.operators):
            if op.kind=="Conv":
                overrides[op.name]=replace(
                    base,pes=2,simd=2,serialization=2 if i%2==0 else 1)
    arch=resolve(model,base,overrides)
    source=emit_hls(lower(arch))
    x=np.asarray([(-1)**i*(i+1)/13 for i in range(16)],
                 dtype=np.float32).reshape(1,1,4,4)
    return arch,source,x


@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
@pytest.mark.parametrize("kind",["branch_concat","pool_fork","broadcast_fork"])
def test_every_observed_graph_intermediate_matches_numpy(kind):
    arch,source,x=setup_source(kind)
    receipt=attest_nchw_intermediates(arch,source,x)
    assert receipt["status"]=="ACTUAL_NCHW_INTERMEDIATE_VALUES_VERIFIED_E1"
    assert receipt["operator_snapshots"]==len(arch.functional.operators)
    assert receipt["observed_tensor_elements"]==sum(
        int(np.prod(arch.functional.tensor(o.outputs[0]).shape))
        for o in arch.functional.operators)
    assert len(receipt["individual_operator_receipts"])==len(arch.functional.operators)
    assert all(r["host_float_max_abs_error"]<1e-3
               for r in receipt["individual_operator_receipts"])
    assert receipt["native_vitis_calls"]==0


@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_numeric_mutant_fails_at_first_affected_intermediate():
    arch,source,x=setup_source("branch_concat")
    matched=re.search(
        r"static const data_t tensor_\d+\[\d+\] = \{data_t\(([^)]*)\)",
        source.cpp)
    assert matched is not None
    altered=source.cpp[:matched.start(1)]+"11.0"+source.cpp[matched.end(1):]
    mutant=Emission(altered,{**source.manifest,
                             "cpp_sha256":sha256(altered.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="SOURCE_NCHW_INTERMEDIATE_NUMERICAL_MISMATCH"):
        attest_nchw_intermediates(arch,mutant,x)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_removed_actual_snapshot_is_not_counted_as_valid_stage_witness():
    arch,source,x=setup_source("pool_fork",different=False)
    altered=source.cpp.replace('FPGAI_TENSOR 0 ', 'BROKEN_TENSOR 0 ')
    assert altered!=source.cpp
    mutant=Emission(altered,{**source.manifest,
                             "cpp_sha256":sha256(altered.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="Unexpected GCC tensor trace"):
        attest_nchw_intermediates(arch,mutant,x)


def test_bounded_missing_source_and_architecture_identity_fail_closed():
    arch,source,x=setup_source("broadcast_fork")
    small=attest_nchw_intermediates(arch,source,x,max_elements=1)
    assert small["status"]=="UNKNOWN_ELEMENT_BUDGET"
    assert small["native_vitis_calls"]==0
    bad=Emission(source.cpp,{**source.manifest,"cpp_sha256":"0"*64})
    with pytest.raises(ValueError,match="digest"):
        attest_nchw_intermediates(arch,bad,x)
    with pytest.raises(ValueError,match="input shape"):
        attest_nchw_intermediates(arch,source,x.ravel())
    other=resolve(arch.functional,ArchitectureRequest())
    with pytest.raises(ValueError,match="Phi"):
        attest_nchw_intermediates(other,source,x)
