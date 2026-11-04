"""Scientific E0/E1 checks of explicit row-block schedules and actual C++.

Counterexamples exercise invalid candidate schedules; valid candidates must
compile, execute and produce the original numerical result. No Vitis calls.
"""
from __future__ import annotations

import shutil
import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.compiler.cst_pipeline import compile_cst
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.cpp_chain_trace import trace_chain_epochs
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi


@pytest.mark.parametrize("rows,block,widths", [
    (3, 1, (3, 5, 4, 2)),
    (3, 2, (3, 5, 4, 2)),
    (5, 2, (3, 5, 4, 2)),
    (5, 3, (3, 5, 4, 3, 2)),
    (5, 5, (3, 5, 4, 3, 2)),
])
def test_actual_blocked_source_implements_certified_schedule(rows, block, widths):
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    graph = make_chain(rows, widths)
    arch = resolve(graph, ArchitectureRequest())
    compiled = compile_row_chain(arch, row_block=block)
    proof = compiled.evidence["schedule_witness"]
    assert proof["status"] == "VERIFIED"
    assert proof["source_scratch_elements"] == block*sum(widths[1:-1])
    assert compiled.emission.manifest["source_intermediate_scratch_elements"] == block*sum(widths[1:-1])
    if block > 1:
        assert f"for (int rb=0; rb<{rows}; rb+={block})" in compiled.emission.cpp
        assert f"scratch_0[ri*{widths[1]}+ni]" in compiled.emission.cpp
    x = np.linspace(-0.125, 0.125, rows*widths[0], dtype=np.float32).reshape(
        rows, widths[0])
    result = validate_host(graph, compiled.emission, x, atol=1e-5, rtol=1e-4)
    assert result["accepted"]
    event_result = trace_chain_epochs(compiled.ir, compiled.emission)
    assert event_result["emitted_row_block"] == block
    assert event_result["producer_ready_before_consumer_finished"]
    if block in (1, 2):
        phi = trace_cpp_phi(compiled.ir.composed.lowered,
                            emission=compiled.emission)
        assert phi["source_phi_agreement"]


def test_identical_phi_produces_different_source_level_temporal_architectures():
    graph = make_chain(5, (3, 5, 4, 2))
    arch = resolve(graph, ArchitectureRequest())
    c1 = compile_row_chain(arch, row_block=1)
    c2 = compile_row_chain(arch, row_block=2)
    c5 = compile_row_chain(arch, row_block=5)
    assert len({c1.emission.manifest["cpp_sha256"],
                c2.emission.manifest["cpp_sha256"],
                c5.emission.manifest["cpp_sha256"]}) == 3
    assert len({c1.emission.manifest["functional_sha256"],
                c2.emission.manifest["functional_sha256"],
                c5.emission.manifest["functional_sha256"]}) == 1
    assert [c.emission.manifest["source_intermediate_scratch_elements"]
            for c in (c1, c2, c5)] == [9, 18, 45]


def test_policy_and_invalid_row_block_fail_closed():
    arch = resolve(make_chain(3, (3, 5, 4, 2)), ArchitectureRequest())
    with pytest.raises(ValueError, match="Row-block scheduling"):
        compile_cst(arch, exchange_policy="whole_tensor", row_block=2)
    with pytest.raises(ValueError, match="row_block|row-block|schedule"):
        compile_cst(arch, exchange_policy="row_chain", row_block=0)
    tested = compile_cst(arch, exchange_policy="row_chain", row_block=2)
    assert tested.emission.manifest["row_block"] == 2
