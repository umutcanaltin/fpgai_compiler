"""Actual restricted branching C++ source and two-reader last-use controls."""
from dataclasses import replace
from hashlib import sha256
import shutil

import numpy as np
import pytest

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.fanout_diamond import (
    emit_fanout_diamond_hls, fanout_diamond_offsets,
)
from fpgai_research.model_ir.contracts import (
    FunctionalModel, Operator, Tensor,
)
from fpgai_research.validation.cpp_fanout_trace import trace_fanout_cpp
from fpgai_research.validation.host import validate_host


def diamond(rows=3, cols=3):
    k=2
    return FunctionalModel(
        tensors=(
            Tensor("x",(rows,k)),
            Tensor("w",(k,cols),data=tuple(
                ((-1.)**i)*(i+1)/8 for i in range(k*cols))),
            Tensor("h",(rows,cols)),Tensor("a",(rows,cols)),
            Tensor("b",(rows,cols)),Tensor("g",(rows,cols)),
            Tensor("out",(rows,cols)),
        ),
        operators=(
            Operator("producer","MatMul",("x","w"),("h",)),
            Operator("reader0","Relu",("h",),("a",)),
            Operator("reader1","Relu",("h",),("b",)),
            Operator("merge","Add",("a","b"),("g",)),
            Operator("final","Relu",("g",),("out",)),
        ),
        inputs=("x",), outputs=("out",),
    )


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC host compiler required")
@pytest.mark.parametrize("rows,cols",[(2,2),(3,3),(5,2)])
def test_actual_diamond_source_and_numerical_reference(rows,cols):
    model=diamond(rows,cols)
    x=np.array([(-1)**i*(i+1)/4 for i in range(rows*2)],
               dtype=np.float32).reshape(rows,2)
    sources=[]
    for reuse in (False,True):
        offsets=fanout_diamond_offsets(model,share_released=reuse)
        emitted=emit_fanout_diamond_hls(model,offsets=offsets)
        result=validate_host(model,emitted,x)
        assert result["accepted"] and result["output_count"]==rows*cols
        trace=trace_fanout_cpp(model,emitted)
        assert trace["status"]=="SOURCE_FANOUT_LIFETIME_VERIFIED"
        assert trace["multi_reader_values"]==rows*cols
        assert trace["observed_events"]==9*rows*cols
        assert trace["native_vitis_calls"]==0
        sources.append(emitted)
    assert sources[0].manifest["scratch_elements"]==4*rows*cols
    assert sources[1].manifest["scratch_elements"]==3*rows*cols
    assert sources[0].manifest["cpp_sha256"]!=sources[1].manifest["cpp_sha256"]


def test_unsafe_early_alias_rejected_before_source_emission():
    model=diamond()
    original=fanout_diamond_offsets(model,share_released=True)
    wrong=dict(original)
    wrong["a"]=wrong["h"]
    with pytest.raises(ValueError,match="SOURCE_ALIAS_UNSAFE"):
        emit_fanout_diamond_hls(model,offsets=wrong)


def test_wrong_graph_and_invalid_numeric_are_never_silently_compiled():
    model=diamond()
    wrong_join=replace(model.operators[3],inputs=("b","a"))
    wrong=replace(model,operators=(
        model.operators[0],model.operators[1],model.operators[2],
        wrong_join,model.operators[4]))
    with pytest.raises(ValueError,match="wiring|sequence|non-topological"):
        emit_fanout_diamond_hls(
            wrong,offsets=fanout_diamond_offsets(model,share_released=False))
    with pytest.raises(ValueError,match="precision"):
        emit_fanout_diamond_hls(
            model,offsets=fanout_diamond_offsets(model,share_released=False),
            precision_bits=6,integer_bits=7)


@pytest.mark.skipif(shutil.which("g++") is None,reason="GCC required")
def test_source_trace_mutation_detects_missing_second_reader():
    model=diamond(2,2)
    emitted=emit_fanout_diamond_hls(
        model,offsets=fanout_diamond_offsets(model,share_released=True))
    damaged=emitted.cpp.replace("FPGAI_FANOUT CONSUME h %d reader1",
                                "FPGAI_FANOUT CONSUME h %d reader0")
    assert damaged!=emitted.cpp
    mutated=Emission(damaged,{
        **emitted.manifest, "cpp_sha256":sha256(damaged.encode()).hexdigest()})
    with pytest.raises(AssertionError,match="SOURCE_FANOUT_EVENT_MISMATCH"):
        trace_fanout_cpp(model,mutated)


def test_source_trace_refuses_tampered_source_digest():
    model=diamond(2,2)
    emitted=emit_fanout_diamond_hls(
        model,offsets=fanout_diamond_offsets(model,share_released=True))
    corrupted=Emission(emitted.cpp+"//tamper\n",emitted.manifest)
    with pytest.raises(ValueError,match="hash"):
        trace_fanout_cpp(model,corrupted)
