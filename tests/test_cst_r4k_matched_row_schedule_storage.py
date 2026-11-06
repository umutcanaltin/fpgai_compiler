"""R4k schedule-controlled MatMul fanout source ablation regression tests.

No GitHub Actions, Vitis, Vivado or physical claims. Optional GCC tests
run actual emitted source, decode every memory access and independently
verify all intermediate values, issue events and final numerical outputs.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import numpy as np
import pytest

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp, row_retained_plan,
)
from fpgai_research.codegen.hls.r4k_full_storage_row_schedule_cpp import (
    emit_full_storage_row_schedule_cpp, full_storage_row_schedule_plan,
)
from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4k_matched_row_schedule_storage as sut
from fpgai_research.model_ir.fanout_contract import admit_fanout
from fpgai_research.validation.r4k_matched_storage_gcc import verify_matched_gcc


@pytest.mark.parametrize("graph", [
    (2, (2, 3, 3, 4), "scalar"),
    (3, (2, 2, 4, 4), "mixed"),
    (4, (3, 2, 3, 4), "mixed"),
    (2, (16, 4, 16, 8), "mixed"),
])
def test_storage_only_control_keeps_original_row_event_and_architecture(graph):
    arch = r4a._case(graph, precision_bits=32, integer_bits=6)
    full = emit_full_storage_row_schedule_cpp(arch)
    row = emit_row_retained_fanout_cpp(arch)
    a = full_storage_row_schedule_plan(arch)
    b = row_retained_plan(arch)
    assert full.manifest["source_cpp_sha256"] == sha256(full.cpp.encode()).hexdigest()
    assert full.manifest["base_row_retained_cpp_sha256"] == (
        row.manifest["source_cpp_sha256"])
    assert a["source_events"] == b["source_events"]
    assert full.manifest["source_event_order_sha256"] == (
        row.manifest["source_event_order_sha256"])
    assert full.manifest["functional_sha256"] == row.manifest["functional_sha256"]
    assert full.manifest["architecture_sha256"] == row.manifest["architecture_sha256"]
    m,k,n,q = graph[1]
    assert full.manifest["source_scratch_array_elements"] - (
        row.manifest["source_scratch_array_elements"]) == (m-1)*n
    assert full.manifest["source_all_declared_mutable_array_elements"] - (
        row.manifest["source_all_declared_mutable_array_elements"]) == (m-1)*n
    assert full.cpp.count(f"const int h_addr=h_idx;") == graph[0]
    assert full.cpp.count(f"const int r4_h_store_addr=h_idx;") == 1
    assert row.cpp.count(f"const int h_addr=h_idx%{n};") == graph[0]
    assert full.manifest["HLS_E3_qualified"] is None
    assert full.manifest["physical_E4_qualified"] is None


def test_r4k_original_32_6_predeclared_only_and_vendor_free():
    p = sut.plan()
    assert p["numeric_format"] == [32, 6]
    assert p["source_scratch_elements_saved"] == 240
    assert p["native_vendor_calls"] == 0
    assert p["physical_E4"] is None
    assert p["matched_source_arms"] == 2
    assert p["clock_ns"] > p["uncertainty_ns"] > 0
    assert len(p["source_schedule_sha256"]) == 64
    assert p["original_R4d_E2_E3_32_6_required"] is True


@pytest.mark.skipif(shutil.which("g++") is None, reason="Real GCC required")
@pytest.mark.parametrize("graph", [
    (2, (2, 3, 3, 4), "scalar"),
    (3, (2, 2, 4, 4), "mixed"),
    (4, (3, 2, 3, 4), "mixed"),
])
def test_real_gcc_both_arms_identical_source_events_and_all_values(graph):
    arch = r4a._case(graph, precision_bits=32, integer_bits=6)
    m,k,n,q=graph[1]
    x=np.asarray([
        [((row*7 + col*3)%11-5)/8.0 for col in range(k)]
        for row in range(m)
    ], dtype=np.float32)
    report=verify_matched_gcc(arch, runtime_input=x)
    assert report["status"] == (
        "MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED")
    assert report["source_scratch_elements_saved"] == (m-1)*n
    assert report["all_intermediates_numerically_equal"] is True
    assert report["all_final_outputs_numerically_equal"] is True
    assert report["original_host_gcc_compilations"] == 2
    assert report["new_native_Vitis_calls"] == 0
    assert report["physical_E4"] is None


def test_absent_original_e2_e3_gate_rejected_before_output(tmp_path, monkeypatch):
    frozen=tmp_path/"old_r4a"
    study=tmp_path/"old_r4d"
    frozen.mkdir()
    study.mkdir()
    monkeypatch.setattr(sut.r4e,"audit_native",
                        lambda **_: {
                            "source_attested_arms": 6,
                            "observations": [
                                {"arm": "full_H", "format": [32,6],
                                 "E2_qualified": True, "E3_qualified": False},
                                {"arm": "row_H", "format": [32,6],
                                 "E2_qualified": True, "E3_qualified": True},
                            ]
                        })
    target=tmp_path/"should_not_create"
    with pytest.raises(ValueError, match="E2/E3 gate"):
        sut.run(original_r4d=study, original_r4a=frozen, output=target)
    assert not target.exists()


@pytest.mark.skipif(shutil.which("g++") is None, reason="Real GCC required")
def test_original_source_preflight_then_two_create_only_packages(tmp_path, monkeypatch):
    config=sut.plan()
    arch=r4a._case(r4a.CASES[3], precision_bits=32, integer_bits=6)
    x=r4a._sample(r4a.CASES[3][1][0], r4a.CASES[3][1][1])
    old=tmp_path/"original_r4a"
    study=tmp_path/"original_r4d"
    old.mkdir()
    p=study/"format_32_06"/"row_H"
    p.mkdir(parents=True)
    emission=emit_row_retained_fanout_cpp(arch)
    (p/"source.cpp").write_text(emission.cpp)
    from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
    write_vitis_package(
        str(p/"vitis_source"), arch.functional,
        r4a._vitis_emission(arch, emission), x,
        part=r4a.PART, clock_ns=r4a.CLOCK_NS,
        clock_uncertainty_ns=r4a.UNCERTAINTY_NS,
        tolerance=r4a.CSIM_ABS_TOLERANCE,
    )
    monkeypatch.setattr(sut.r4e,"audit_native",
                        lambda **_: {
                            "source_revision": "a"*40,
                            "original_R4a_source_revision": "b"*40,
                            "source_attested_arms": 6,
                            "observations": [
                                {"arm": arm, "format": [32,6],
                                 "E2_qualified": True, "E3_qualified": True}
                                for arm in ("full_H","row_H")
                            ],
                        })
    monkeypatch.setattr(sut.r4a,"_git_revision",lambda: "c"*40)
    target=tmp_path/"new_r4k"
    result=sut.run(original_r4d=study, original_r4a=old, output=target)
    assert result["source_only_E1_verified"] is True
    assert result["E1"]["source_scratch_elements_saved"] == 240
    assert result["new_vendor_Vitis_calls"] == 0
    assert result["HLS_E2_run"] is False
    assert result["HLS_E3_run"] is False
    assert (target/"full_storage.host_stdout.txt").is_file()
    assert (target/"row_storage.host_stdout.txt").is_file()
    assert (target/"full_storage"/"vitis_source"/"model.cpp").is_file()
    assert (target/"row_storage"/"vitis_source"/"tb.cpp").is_file()
    inventory=json.loads((target/"artifact_sha256.json").read_text())
    assert inventory["summary.json"] == sha256(
        (target/"summary.json").read_bytes()).hexdigest()
    assert inventory["full_storage.host_stdout.txt"] == sha256(
        (target/"full_storage.host_stdout.txt").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(original_r4d=study, original_r4a=old, output=target)
    assert sha256((p/"source.cpp").read_bytes()).hexdigest() == (
        config["source_row_original_sha256"])


@pytest.mark.skipif(shutil.which("g++") is None, reason="Real GCC required")
def test_actual_h_load_mutation_detected_even_when_trace_witness_unchanged(
    monkeypatch,
):
    """A forged address witness alone is not sufficient for numerical safety."""
    from fpgai_research.validation import r4k_matched_storage_gcc as validation
    arch=r4a._case((2,(2,2,3,4),"scalar"),precision_bits=32,integer_bits=6)
    actual=emit_full_storage_row_schedule_cpp(arch)
    old="      data_t h_val=scratch[h_addr];"
    assert actual.cpp.count(old) == 2
    modified=actual.cpp.replace(
        old, "      data_t h_val=scratch[(h_addr+1)%16];", 1)
    forged=Emission(
        modified,
        {**actual.manifest,
         "source_cpp_sha256":sha256(modified.encode()).hexdigest()},
    )
    monkeypatch.setattr(validation, "emit_full_storage_row_schedule_cpp",
                        lambda _arch: forged)
    x=np.asarray([[0.3,-0.5],[-0.7,1.1]],dtype=np.float32)
    with pytest.raises(AssertionError, match="numerical mismatch"):
        validation.verify_matched_gcc(arch,runtime_input=x)
