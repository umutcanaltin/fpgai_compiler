"""R4w one-time new Vivado clock sweep: synthetic source/ROM fixture tests.

Fixtures exercise preparation, exact source copies, strict candidate gates
and immutable attempt prevention; NO actual vendor tools are launched.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path

import pytest

from fpgai_research.experiments import (
    cst_paper_r4w_feasible_clock_sweep as sut,
)


def _fixture(tmp_path,monkeypatch):
    root=tmp_path/"r4o"
    r4k=tmp_path/"r4k"
    r4d=tmp_path/"r4d"
    r4a=tmp_path/"r4a"
    for x in (root,r4k,r4d,r4a):
        x.mkdir()
    cases={}
    for arm in sut.ARMS:
        folder=root/arm/"rtl"
        folder.mkdir(parents=True)
        cpp=b"// SHA-pinned synthetic C++ model\n"
        rtl=f"module {arm}; endmodule\n".encode()
        rom=(arm+" test rom\n").encode()
        (folder/"kernel.v").write_bytes(rtl)
        (folder/"weight.dat").write_bytes(rom)
        cases[arm]={
            "rtl_sha256":{"kernel.v":sha256(rtl).hexdigest()},
            "rom_sha256":{"weight.dat":sha256(rom).hexdigest()},
            "source_cpp_sha256":sha256(cpp).hexdigest(),
        }
    fake={
        "schema":"synthetic original R4o for tests only",
        "original_R4k_source_revision":"7"*40,
        "target_part":sut.r4o.PART,
        "arms":cases,
    }
    monkeypatch.setattr(sut,"_origins",lambda **kwargs:fake)
    return dict(r4o_study=root,r4k_study=r4k,original_r4d=r4d,
                original_r4a=r4a),fake


def test_prepare_audits_all_six_sha_pinned_original_cases(tmp_path,monkeypatch):
    kwargs,fake=_fixture(tmp_path,monkeypatch)
    out=tmp_path/"new_r4w"
    report=sut.prepare(**kwargs,output=out)
    assert report["new_vendor_calls"]==0
    assert report["original_evidence_modified"] is False
    assert len(report["predeclared_cases"])==6
    assert sut.audit_prepared(study=out,**kwargs)==report
    cases=report["predeclared_cases"]
    assert cases["full_storage_period_16ns"]["period_ns"]==16.0
    assert cases["row_storage_period_24ns"]["period_ns"]==24.0
    for arm in sut.ARMS:
        for period in sut.PERIODS_NS:
            case=out/sut._case(arm,period)/"vivado"
            script=(case/"run_vivado.tcl").read_text()
            assert f"-period {period:.2f}" in script
            assert "set_clock_uncertainty 1.50" in script
            assert "report_drc" in script
            assert "report_timing_summary" in script
            assert "route_design" in script
            assert "write_checkpoint" in script
            assert (case/"rtl"/"kernel.v").read_bytes()==(
                kwargs["r4o_study"]/arm/"rtl"/"kernel.v").read_bytes()
            assert (case/"weight.dat").read_bytes()==(
                kwargs["r4o_study"]/arm/"rtl"/"weight.dat").read_bytes()
    with pytest.raises(FileExistsError,match="create-only"):
        sut.prepare(**kwargs,output=out)


def test_prepared_source_drift_rejected_with_original_unchanged(tmp_path,monkeypatch):
    opts,_=_fixture(tmp_path,monkeypatch)
    out=tmp_path/"new_r4w"
    sut.prepare(**opts,output=out)
    orig=opts["r4o_study"]/"row_storage"/"rtl"/"kernel.v"
    orig_hash=sut._sha(orig)
    staged=out/"row_storage_period_20ns"/"vivado"/"rtl"/"kernel.v"
    staged.write_text("tampered Verilog")
    with pytest.raises(ValueError,match="staged source"):
        sut.audit_prepared(study=out,**opts)
    assert sut._sha(orig)==orig_hash


def test_prepared_tcl_constraint_drift_rejected(tmp_path,monkeypatch):
    opts,_=_fixture(tmp_path,monkeypatch)
    out=tmp_path/"new_r4w"
    sut.prepare(**opts,output=out)
    script=out/"full_storage_period_16ns"/"vivado"/"run_vivado.tcl"
    script.write_text(script.read_text().replace("-period 16.00",
                                                 "-period 5.00"))
    with pytest.raises(ValueError,match="script or clock"):
        sut.audit_prepared(study=out,**opts)


def test_dry_run_starts_zero_vendor_calls_and_audit_rejects_missing_native(
        tmp_path,monkeypatch):
    opts,_=_fixture(tmp_path,monkeypatch)
    out=tmp_path/"new_r4w"
    sut.prepare(**opts,output=out)
    report=sut.execute(study=out,**opts)
    assert report["status"]=="R4W_DRY_RUN_NO_VENDOR_CALLS"
    assert report["planned_new_vivado_calls"]==6
    assert report["actual_new_vivado_calls"]==0
    audit=sut.audit_native(study=out,**opts)
    assert audit["qualified_pair_count"]==0
    assert audit["native_attempt_lock_verified"] is False
    assert audit["all_original_probes_attempted"] is False
    assert all(not c["strict_clean_OOC_qualified"]
               for c in audit["cases"].values())


def test_native_requires_two_explicit_opt_ins_and_never_retries(
        tmp_path,monkeypatch):
    opts,_=_fixture(tmp_path,monkeypatch)
    out=tmp_path/"new_r4w"
    sut.prepare(**opts,output=out)
    with pytest.raises(ValueError,match="two explicit"):
        sut.execute(study=out,**opts,execute_native=True)
    # The new attempt lock can never be deleted or auto-retried.
    lock=out/"r4w_one_time_native_attempt_lock.json"
    lock.write_text('{"preserve_original":true}\n')
    with pytest.raises(FileExistsError,match="locked"):
        sut.execute(study=out,**opts,
                    execute_native=True,confirm_isolated_new_native=True,
                    vivado_executable="/bin/true")


def test_new_study_output_cannot_nest_inside_original_roots(
        tmp_path,monkeypatch):
    opts,_=_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="outside"):
        sut.prepare(**opts,output=opts["r4o_study"]/"unsafe_new_result")


def test_missing_original_stage_or_invalid_clocks_fail_closed():
    with pytest.raises(ValueError,match="unregistered"):
        sut._case("fake_arm",16.0)
    with pytest.raises(ValueError,match="unregistered"):
        sut._case("row_storage",18.0)
    with pytest.raises(ValueError,match="unsafe HDL"):
        sut._tcl("row_storage",16.0,["../hijack.v"])
