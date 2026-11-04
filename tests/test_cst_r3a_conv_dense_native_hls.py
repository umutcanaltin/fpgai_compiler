"""R3a new opt-in native CNN/Dense HLS checks: NO tools run in tests.

All six-case original native receipts are synthetic within this test;
only actual vendor execution in user's original source folder may
establish real E2/E3. Dry-run and contrast audits are read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpgai_research.experiments import cst_r3a_conv_dense_native_hls as sut


def _source_fixture(monkeypatch,tmp_path):
    root=tmp_path/"new_neural_original"
    root.mkdir()
    rows=[]
    for i in range(6):
        network=("cnn_pool_dense","cnn_strided_dense",
                 "cnn_two_conv_softmax")[i//2]
        row={
            "case":f"case_{i:03d}",
            "network":network,
            "independent_phi":bool(i%2),
            "functional_sha256":f"{i//2+1:064x}",
            "original_input_batch_sha256":f"{i//2+4:064x}",
            "prepared_vitis_testbench_sha256":f"{i//2+7:064x}",
            "prepared_vitis_tcl_sha256":f"{i//2+11:064x}",
            "source_cpp_sha256":f"{i+16:064x}",
        }
        rows.append(row)
        case=root/row["case"]
        case.mkdir()
    (root/"cases.json").write_text(json.dumps(rows))
    verified={
        "source_revision":"a"*40,
        "original_source_cases":6,
        "original_host_GCC_qualified_input_receipts":12,
        "original_source_packages_SHA_attested":6,
    }
    monkeypatch.setattr(sut,"audit_source",lambda *,study:verified)
    # Existing strict native qualifier and source parser mocked, so
    # tests check orchestration without manufacturing native evidence.
    def source_summary(folder):
        case=Path(folder).name
        return {"source_sha256":next(r["source_cpp_sha256"] for r in rows
                                 if r["case"]==case)}
    monkeypatch.setattr(sut,"summarize_case",source_summary)
    def qualified(case,expected_case,*,summary):
        return {
            "native_e2_accepted":False,
            "native_e3_qualified":False,
            "status":"UNKNOWN_NO_ORIGINAL_VITIS",
            "evidence_level":"E1-HLS-source",
            "hls_worst_cycles":None,
            "hls_scheduling_margin_ns":None,
            "hls_estimated_clock_ns":None,
            "estimated_resources":{k:None for k in sut.METRICS},
            "hls_report_sha256":None,
            "warnings":[],
        }
    monkeypatch.setattr(sut,"qualify_case",qualified)
    return root,rows


def test_r3a_prespecified_three_source_matched_networks():
    assert sut.SOURCE_CASES==6
    assert sut.HOST_INPUTS_PER_CASE==2
    assert sut.PAIRS==((0,1),(2,3),(4,5))
    assert sut.CLOCK_NS==10.0
    assert sut.UNCERTAINTY_NS==1.0


def test_read_only_source_hls_audit_unknown_is_never_zero_or_tie(
        monkeypatch,tmp_path):
    root,_=_source_fixture(monkeypatch,tmp_path)
    audited=sut.audit(study=root)
    assert audited["original_source_cases"]==6
    assert audited["original_GCC_source_inputs_qualified"]==12
    assert audited["native_CSim_E2_qualified_cases"]==0
    assert audited["native_CSynth_E3_qualified_cases"]==0
    assert audited["original_vitis_invocation_receipts"]==0
    assert audited["new_original_Vitis_calls_during_audit"]==0
    assert all(x["HLS_estimated_resources"]["bram_18k"] is None
               for x in audited["cases"])
    comparisons=sut.contrasts(study=root)
    assert comparisons["prespecified_original_neural_pairs"]==3
    assert comparisons["E3_both_qualified_pairs"]==0
    assert comparisons["E3_unknown_or_rejected_pairs"]==3
    assert all(x["HLS_estimated_BRAM18k_delta"] is None
               for x in comparisons["pairs"])
    assert all(x["status"]=="UNKNOWN_NOT_BOTH_ORIGINAL_E3"
               for x in comparisons["pairs"])


def test_dryrun_with_no_native_is_side_effect_free(monkeypatch,tmp_path):
    root,_=_source_fixture(monkeypatch,tmp_path)
    before={p.name:p.read_bytes() for p in root.rglob("*.json")}
    res=sut.execute(study=root,start_case=2,max_cases=2)
    assert res["status"]=="DRY_RUN_NO_VITIS_CALLS"
    assert res["selected_original_cases"]==["case_002","case_003"]
    assert res["original_vitis_calls"]==0
    after={p.name:p.read_bytes() for p in root.rglob("*.json")}
    assert before==after


def test_native_requires_two_flags_selection_and_available_tool(
        monkeypatch,tmp_path):
    root,_=_source_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match="explicit confirmation"):
        sut.execute(study=root,execute_native=True)
    with pytest.raises(ValueError,match="0..5"):
        sut.execute(study=root,start_case=5,max_cases=2)
    with pytest.raises(ValueError,match="0..5"):
        sut.execute(study=root,start_case=True,max_cases=1)
    with pytest.raises(ValueError,match="time budget"):
        sut.execute(study=root,execute_native=True,
                    confirm_isolated_native=True,timeout_seconds=0)
    monkeypatch.setattr(sut.shutil,"which",lambda _:None)
    with pytest.raises(FileNotFoundError,match="executable unavailable"):
        sut.execute(study=root,execute_native=True,
                    confirm_isolated_native=True)


def test_original_native_attempt_cannot_overwrite_receipt(
        monkeypatch,tmp_path):
    root,_=_source_fixture(monkeypatch,tmp_path)
    (root/"case_000"/"vitis_invocation.json").write_text('{"tool":"vitis_hls"}')
    monkeypatch.setattr(sut.shutil,"which",lambda _:"/fake/vitis_hls")
    with pytest.raises(FileExistsError,match="already attempted"):
        sut.execute(study=root,execute_native=True,
                    confirm_isolated_native=True)


def test_source_pair_mismatch_rejected_before_read_only_audit(
        monkeypatch,tmp_path):
    root,rows=_source_fixture(monkeypatch,tmp_path)
    rows[1]["prepared_vitis_tcl_sha256"]="f"*64
    (root/"cases.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError,match="controls do not match"):
        sut.audit(study=root)


def test_qualified_hls_comparisons_are_estimates_only(
        monkeypatch,tmp_path):
    root,_=_source_fixture(monkeypatch,tmp_path)
    def measured(case,expected_case,*,summary):
        i=int(Path(case).name.split("_")[1])
        return {
            "native_e2_accepted":True,"native_e3_qualified":i<2,
            "status":("QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL" if i<2
                      else "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"),
            "evidence_level":"E3-HLS-synthesis-report",
            "hls_worst_cycles":100+i,
            "hls_scheduling_margin_ns":1.0,
            "hls_estimated_clock_ns":9.0,
            "estimated_resources":{
                "bram_18k":3+i,
                "lut":100+i,"ff":200+i,"dsp":5,"uram":0},
            "hls_report_sha256":f"{i+100:064x}",
            "warnings":[],
        }
    monkeypatch.setattr(sut,"qualify_case",measured)
    pairs=sut.contrasts(study=root)
    assert pairs["E3_both_qualified_pairs"]==1
    assert pairs["E3_unknown_or_rejected_pairs"]==2
    first=pairs["pairs"][0]
    assert first["HLS_estimated_BRAM18k_delta"]==1
    assert first["HLS_worst_latency_cycles_delta"]==1
    assert first["physical_BRAM_delta"] is None
    assert all(x["HLS_estimated_BRAM18k_delta"] is None
               for x in pairs["pairs"][1:])



def test_original_e3_clock_failure_keeps_diagnostic_but_no_qor(monkeypatch, tmp_path):
    """A failed effective clock is source-attested E3 evidence, not E3 QoR."""
    root, _ = _source_fixture(monkeypatch, tmp_path)
    (root / "case_002" / "vitis_invocation.json").write_text("{}")

    def qualified(case, expected_case, *, summary):
        i = int(Path(case).name.split("_")[1])
        failed = i == 2
        return {
            "native_e2_accepted": i < 3,
            "native_e3_qualified": i < 2,
            "status": ("FAILED_HLS_EFFECTIVE_CLOCK_BUDGET" if failed
                       else ("QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL" if i < 2
                             else "UNKNOWN_NO_ORIGINAL_VITIS")),
            "evidence_level": "E3-HLS-synthesis-report" if i < 3 else "E1-HLS-source",
            "hls_timing_feasible": (not failed if i < 3 else None),
            "hls_worst_cycles": 624 + i,
            "hls_estimated_clock_ns": 9.31 if failed else 8.98,
            "hls_effective_scheduling_budget_ns": 9.0,
            "hls_scheduling_margin_ns": -0.31 if failed else 0.02,
            "estimated_resources": {
                "bram_18k": 7, "dsp": 11, "lut": 100, "ff": 200, "uram": 0},
            "hls_report_sha256": "f" * 64,
            "warnings": [],
        }

    monkeypatch.setattr(sut, "qualify_case", qualified)
    cases = sut.audit(study=root)["cases"]
    failed = cases[2]
    assert failed["native_CSim_E2_accepted"] is True
    assert failed["native_CSynth_E3_qualified"] is False
    assert failed["qualification_status"] == "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
    assert failed["HLS_estimated_resources"]["bram_18k"] is None
    assert failed["hls_worst_cycles"] is None
    assert failed["hls_estimated_clock_ns"] is None
    assert failed["attested_HLS_timing_diagnostic"]["estimated_clock_ns"] == 9.31
    assert failed["attested_HLS_timing_diagnostic"]["effective_budget_ns"] == 9.0
    assert failed["attested_HLS_timing_diagnostic"]["feasible"] is False
    assert cases[3]["attested_HLS_timing_diagnostic"] is None
    matched = sut.contrasts(study=root)["pairs"]
    assert matched[1]["status"] == "REJECTED_ORIGINAL_E3_ARM"
    assert matched[1]["HLS_estimated_BRAM18k_delta"] is None
    assert matched[1]["HLS_worst_latency_cycles_delta"] is None
