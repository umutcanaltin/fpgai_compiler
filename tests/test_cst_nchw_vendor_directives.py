"""Vendor-specific Vitis HLS pragma intent is real source, not FPGA truth."""
from dataclasses import replace
from hashlib import sha256
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.nchw_shared_scratch import (
    emit_nchw_stage_scratch,greedy_nchw_stage_plan,
)
from fpgai_research.codegen.hls.nchw_vendor_directives import (
    PRAGMA,apply_native_scratch_binding,
)
from fpgai_research.experiments.cst_nchw_graph_fixtures import nchw_case
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.experiments.cst_nchw_native_pairs import prepare,audit_prepared


def test_native_bram_2p_binding_is_source_attested_but_not_hardware_proof():
    model=nchw_case("pool_fork")
    arch=resolve(model,ArchitectureRequest())
    offsets=greedy_nchw_stage_plan(arch,reuse=True)
    emitted=emit_nchw_stage_scratch(arch,offsets)
    automatic=apply_native_scratch_binding(emitted,policy="auto")
    explicit=apply_native_scratch_binding(emitted,policy="bram_2p")
    assert PRAGMA not in automatic.cpp
    assert explicit.cpp.count(PRAGMA)==1
    assert explicit.cpp!=automatic.cpp
    assert explicit.manifest["requested_native_memory_binding"]=="bram_2p"
    assert explicit.manifest["requested_native_pragma"]==PRAGMA
    assert explicit.manifest["binding_physically_verified"] is False
    assert explicit.manifest["native_vitis_calls"]==0
    assert explicit.manifest["cpp_sha256"]==sha256(explicit.cpp.encode()).hexdigest()
    with pytest.raises(ValueError,match="duplicated"):
        apply_native_scratch_binding(explicit,policy="bram_2p")
    with pytest.raises(ValueError,match="polic"):
        apply_native_scratch_binding(emitted,policy="lattice")
    with pytest.raises(ValueError,match="fingerprint|source-attested|manifest"):
        altered=Emission(emitted.cpp+"//tampered\n",emitted.manifest)
        apply_native_scratch_binding(altered,policy="bram_2p")


def test_native_pairs_with_same_explicit_vendor_directive(tmp_path):
    study=tmp_path/"bram_policy"
    summary=prepare(output=study,scratch_binding="bram_2p")
    assert summary["case_count"]==12
    assert summary["requested_native_memory_binding"]=="bram_2p"
    assert summary["original_vitis_calls"]==0
    assert summary["native_qualified_cases"]==0
    for first,second in (summary["cases"][i:i+2]
                         for i in range(0,12,2)):
        a=(study/first["case"]/"model.cpp").read_text()
        b=(study/second["case"]/"model.cpp").read_text()
        assert a.count(PRAGMA)==b.count(PRAGMA)==1
        assert first["source_cpp_sha256"]!=second["source_cpp_sha256"]
        assert first["tcl_sha256"]==second["tcl_sha256"]
        assert first["testbench_sha256"]==second["testbench_sha256"]
        assert first["real_scratch_elements"]>second["real_scratch_elements"]
    assert audit_prepared(study)["status"]=="SOURCE_ONLY_PACKAGES_VERIFIED"
    p=study/"case_003"/"model.cpp"
    p.write_text(p.read_text().replace(PRAGMA,"// pragma removed"))
    with pytest.raises(ValueError,match="Tampered|pragma"):
        audit_prepared(study)
