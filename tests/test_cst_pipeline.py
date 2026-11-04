"""CST policy API: same graph and local Phi, three real HLS schedules."""
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.compiler.cst_pipeline import (
    SUPPORTED_EXCHANGE_POLICIES, compile_cst,
)
from fpgai_research.experiments.cst_row_campaign import model_for
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower


def test_three_policies_retain_one_functional_graph_and_diverge_in_real_cpp():
    model = model_for(3, 3, 5, 2)
    arch = resolve(model, ArchitectureRequest())
    default = compile_cst(arch)
    changed = compile_cst(
        arch, exchange_policy="column_major", intermediate_tensor="h")
    row = compile_cst(arch, exchange_policy="row_lifetime")
    assert tuple(SUPPORTED_EXCHANGE_POLICIES) == (
        "whole_tensor", "column_major", "row_lifetime", "row_chain")
    assert default.emission.cpp == emit_hls(lower(arch)).cpp
    assert len({
        default.emission.manifest["cpp_sha256"],
        changed.emission.manifest["cpp_sha256"],
        row.emission.manifest["cpp_sha256"],
    }) == 3
    assert len({
        default.emission.manifest["functional_sha256"],
        changed.emission.manifest["functional_sha256"],
        row.emission.manifest["functional_sha256"],
    }) == 1
    assert default.policy == "whole_tensor"
    assert changed.policy == "column_major"
    assert row.policy == "row_lifetime"
    assert row.emission.manifest["intermediate_scratch_elements"] == 5
    chain = compile_cst(arch, exchange_policy="row_chain")
    assert chain.ir.physical_scratch_elements == 5
    assert chain.emission.manifest["functional_sha256"] == default.emission.manifest["functional_sha256"]


def test_cst_policy_unknown_and_unsupported_modes_fail_closed():
    arch = resolve(model_for(3, 3, 5, 2), ArchitectureRequest())
    with pytest.raises(ValueError, match="Unsupported CST"):
        compile_cst(arch, exchange_policy="stream")
    with pytest.raises(ValueError, match="requires a tensor"):
        compile_cst(arch, exchange_policy="column_major")
    with pytest.raises(ValueError, match="cannot request"):
        compile_cst(arch, intermediate_tensor="h")
    with pytest.raises(ValueError, match="derives its intermediate"):
        compile_cst(arch, exchange_policy="row_lifetime",
                    intermediate_tensor="h")
    with pytest.raises(ValueError, match="internal produced"):
        compile_cst(arch, exchange_policy="column_major",
                    intermediate_tensor="x")
