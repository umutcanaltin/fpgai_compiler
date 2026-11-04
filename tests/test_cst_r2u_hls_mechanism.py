"""R2u full factorial source/host correctness; native measurements are separate.

No Vitis process is invoked in pytest, no GitHub Actions or RL changes.
The original R2h GCC checker validates numeric values/actual source
events of multiple transformed C++ variants against the same E0 map.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp,
)
from fpgai_research.experiments import cst_r2u_hls_mechanism_prepare as prep
from fpgai_research.experiments import cst_r2u_hls_mechanism_execute as native
from fpgai_research.experiments.cst_r2b_source_campaign import _input
from fpgai_research.validation.r2h_allocated_fanout_cpp import verify_allocated_gcc


def test_full_factorial_is_unique_and_bounded():
    assert len(prep.GRAPHS)==2
    assert prep.CASES==32
    assert len(prep.ARMS)==16
    assert len({tuple(arm) for arm in prep.ARMS})==16
    assert all(len(prep.FACTORS[name])==2 for name in prep.FACTOR_NAMES)
    assert prep.CLOCK_NS==8.0
    assert prep.UNCERTAINTY_NS==1.5
    assert prep.TOLERANCE==0.01
    assert set(prep.FACTORS)=={
        "storage","address","binding","padding"
    }


@pytest.mark.parametrize("graph_index",(0,1))
def test_r2u_full_source_variation_in_each_factor(graph_index):
    arch=prep._arch(graph_index)
    rows=[]
    for arm in prep.ARMS:
        named=dict(zip(prep.FACTOR_NAMES,arm))
        emission=prep.emit_factorial_arm(arch,**named)
        source=emission.cpp
        assert sha256(source.encode()).hexdigest()==emission.manifest[
            "cpp_sha256"]
        assert emission.manifest["experimental_address_lookup"]==named["address"]
        assert emission.manifest["experimental_memory_binding"]==named["binding"]
        assert emission.manifest["experimental_array_padding"]==named["padding"]
        assert emission.manifest["source_logical_scratch_slots"]<=emission.manifest[
            "source_scratch_array_elements"]
        assert emission.manifest["source_pad_unused_declared_elements"]>=0
        assert emission.manifest["physical_memory_measurements"] is None
        if named["binding"]=="ram_2p_bram":
            assert "#pragma HLS bind_storage variable=scratch type=ram_2p impl=bram" in source
        else:
            assert "#pragma HLS bind_storage" not in source
        if named["address"]=="switch":
            assert "switch(logical)" in source
            assert "static const int r2h_slot_" not in source
            assert "r2u_lookup_" in source
        else:
            assert "static const int r2h_slot_" in source
            assert "r2u_lookup_" not in source
        if named["padding"]=="pad64":
            assert emission.manifest["source_scratch_array_elements"]%64==0
        rows.append((arm,emission.manifest["cpp_sha256"]))
    assert len({row[1] for row in rows})==len(prep.ARMS)


@pytest.mark.skipif(shutil.which("g++") is None,reason="Real GNU GCC required")
@pytest.mark.parametrize("address,binding,padding",(
    ("table","auto","exact"),
    ("switch","auto","exact"),
    ("table","ram_2p_bram","pad64"),
    ("switch","ram_2p_bram","pad64"),
))
def test_actual_generated_host_cpp_interventions_preserve_source_and_numeric_E1(
        address,binding,padding):
    arch=prep._arch(0)
    m,k=prep.GRAPHS[0][1][:2]
    for storage in prep.FACTORS["storage"]:
        em=prep.emit_factorial_arm(
            arch,storage=storage,address=address,
            binding=binding,padding=padding)
        baseline=emit_allocated_fanout_cpp(arch,layout=storage)
        # The legacy R2h checker preflight requires the original exact
        # logical scratch capacity; retain that original certified map
        # when verifying transformed source. Its GCC executes ACTUAL
        # modified C++ and compares every traced address/numeric value.
        receipt_manifest={
            **baseline.manifest,
            "source_cpp_sha256":sha256(em.cpp.encode()).hexdigest()
        }
        verified=verify_allocated_gcc(
            arch,Emission(em.cpp,receipt_manifest),
            runtime_input=_input(m,k))
        assert verified["status"]=="R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
        assert verified["source_cpp_sha256"]==sha256(em.cpp.encode()).hexdigest()


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    target=tmp_path_factory.mktemp("r2u_hls_factorial")/"original"
    summary=prep.prepare(output=target)
    return target,summary


def test_all_32_original_vitis_packages_matched_and_source_attested(packages):
    root,summary=packages
    assert summary["source_prepared_cases"]==32
    assert summary["arms_per_graph"]==16
    assert summary["source_only_no_new_vitis_calls"] is True
    assert summary["native_csynth_qualified_cases"]==0
    assert summary["physical_routed_BRAM"] is None
    audited=prep.audit_prepared(study=root)
    assert audited["attested_source_cases"]==32
    cases=json.loads((root/"factorial_source_manifest.json").read_text())["rows"]
    assert len(cases)==32
    for gi in range(2):
        rows=cases[16*gi:16*(gi+1)]
        assert len(set(r["testbench_sha256"] for r in rows))==1
        assert len(set(r["tcl_sha256"] for r in rows))==1
        assert len(set(r["numeric_input_sha256"] for r in rows))==1
        assert len(set(r["source_cpp_sha256"] for r in rows))==16
    with pytest.raises(FileExistsError,match="create-only"):
        prep.prepare(output=root)


def test_zero_vitis_audit_must_report_all_32_unknown_and_64_unknown_pairs(packages):
    root,summary=packages
    native_results=native.audit_native(study=root)
    assert native_results["planned_source_cases"]==32
    assert native_results["native_CSim_E2_qualified_cases"]==0
    assert native_results["native_CSynth_E3_qualified_cases"]==0
    assert native_results["prespecified_matched_factor_contrasts"]==64
    assert native_results["E3_both_qualified_contrasts"]==0
    assert native_results["E3_UNKNOWN_or_rejected_contrasts"]==64
    assert native_results["physical_BRAM_measurements"] is None
    assert native_results["original_vendor_calls_during_audit"]==0


def test_native_default_dry_run_and_execution_without_confirmation_refuses(packages):
    root,summary=packages
    dry=native.execute(study=root,start_case=0,max_cases=2)
    assert dry["status"]=="DRY_RUN_NO_NATIVE_CALLS"
    assert dry["original_Vitis_calls"]==0
    assert dry["cases"]==["case_000","case_001"]
    with pytest.raises(ValueError,match="confirmation"):
        native.execute(
            study=root,start_case=0,max_cases=1,
            execute_native=True,confirm_isolated_native=False)
    with pytest.raises(ValueError,match="0..31"):
        native.execute(study=root,start_case=31,max_cases=2)
    with pytest.raises(ValueError,match="finite"):
        prep.prepare(output=root.parent/"badclock",clock_ns=5.0,
                     uncertainty_ns=5.1)


def test_all_four_controlled_contrasts_and_no_imputed_missing_hardware():
    rows=[]
    for gi in range(2):
        for i,arm in enumerate(prep.ARMS):
            row={**dict(zip(prep.FACTOR_NAMES,arm)),
                 "graph_index":gi,"case":f"case_{gi*16+i:03d}",
                 "original_E2_qualified":True,
                 "original_E3_qualified":True,
                 "estimated_resources":{
                     key:((i+1) if key=="bram_18k" else 10)
                     for key in native.RESOURCES},
                 "hls_worst_cycles":200+i}
            rows.append(row)
    for factor in prep.FACTOR_NAMES:
        contrasts=native._contrast(rows,factor)
        assert len(contrasts)==16
        assert all(c["matched_original_CSynth_qualified"] for c in contrasts)
        assert all(c["estimated_resource_delta_b_minus_a"]["bram_18k"]
                   is not None for c in contrasts)
        assert all(c["physical_resource_delta"] is None for c in contrasts)
    rows[0]["original_E3_qualified"]=False
    for factor in prep.FACTOR_NAMES:
        matches=native._contrast(rows,factor)
        assert sum(c["status"]=="UNKNOWN_NOT_BOTH_E3_QUALIFIED"
                   for c in matches)==1
        assert all(c["estimated_resource_delta_b_minus_a"]["bram_18k"] is None
                   for c in matches if not c["matched_original_CSynth_qualified"])


def test_tampered_hls_source_and_numerical_testbench_rejected(packages,tmp_path):
    root,_=packages
    destination=tmp_path/"copy"
    shutil.copytree(root,destination)
    model=destination/"case_004"/"model.cpp"
    model.write_text(model.read_text()+"\n// forged HLS source\n")
    with pytest.raises(ValueError,match="SHA mismatch"):
        prep.audit_prepared(study=destination)
    model.write_bytes((root/"case_004"/"model.cpp").read_bytes())
    tb=destination/"case_015"/"tb.cpp"
    tb.write_text(tb.read_text()+"\n// different numeric control\n")
    with pytest.raises(ValueError,match="SHA mismatch"):
        prep.audit_prepared(study=destination)
