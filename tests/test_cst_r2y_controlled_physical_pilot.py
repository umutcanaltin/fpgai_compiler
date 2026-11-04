"""R2y controlled 9-case Vitis -> Vivado physical study smoke tests.

All original RTL, vendor receipts and test tool runs in this suite are
synthetic. Never call Vivado or Vitis inside pytest.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r2y_controlled_physical_pilot as study
from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl


def _sha(data:bytes)->str:
    return sha256(data).hexdigest()


def _synthetic_hls_source(monkeypatch,tmp_path):
    original=tmp_path/"original_hls"
    original.mkdir()
    rows=[None]*32
    observed=[None]*32
    for i in range(32):
        case=f"case_{i:03d}"
        # Original R2u binary factorial ordering:
        # storage *8, address *4, binding *2, padding *1.
        idx=i%16
        level={
            "storage":"disjoint" if idx<8 else "per_element_optimal",
            "address":"table" if idx%8<4 else "switch",
            "binding":"auto" if idx%4<2 else "ram_2p_bram",
            "padding":"exact" if idx%2==0 else "pad64",
        }
        graph=i//16
        row={
            "case":case,"graph_index":graph,
            "functional_sha256":f"{graph+1:064x}",
            "architecture_sha256":f"{graph+2:064x}",
            "numeric_input_sha256":f"{graph+3:064x}",
            "testbench_sha256":f"{graph+4:064x}",
            "tcl_sha256":f"{graph+5:064x}",
            **level,
        }
        rows[i]=row
        observed[i]={
            "case":case,"original_E3_qualified":i not in (4,12),
            "estimated_resources":{"bram_18k":i%4},
        }
        if i not in study.ORIGINAL_CASE_INDICES:
            continue
        folder=original/case
        rtl=folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        rtl.mkdir(parents=True)
        text=f"module fpgai_model (input ap_clk); endmodule // {i}\n"
        (rtl/"fpgai_model.v").write_text(text)
        (rtl/"original_rom.dat").write_bytes(f"ROM_{i}".encode())
        (folder/"model.cpp").write_text("int source_"+str(i)+";\n")
        (folder/"tb.cpp").write_text("testbench_graph_"+str(graph)+"\n")
        (folder/"manifest.json").write_text(json.dumps({"case":case}))
        row.update({
            "source_cpp_sha256":study._sha(folder/"model.cpp"),
            "testbench_sha256":study._sha(folder/"tb.cpp"),
            "case_manifest_sha256":study._sha(folder/"manifest.json"),
        })
        inv=inventory_vitis_rtl(folder)
        (folder/"vitis_invocation.json").write_text(json.dumps({
            "schema":"fpgai-research/vendor-invocation/v1",
            "tool":"vitis_hls","completed_successfully":True,
            "inputs_stable_during_execution":True,
            "source_sha256":row["source_cpp_sha256"],
            "rtl_output_sha256":inv["rtl_sha256"],
            "rom_output_sha256":inv["rom_sha256"],
        }))
    cohort={
        "source_revision":study.ORIGINAL_R2U_SOURCE_REVISION,
        "clock_ns":study.EXPECTED_CLOCK_NS,
        "part":"xck26-sfvc784-2LV-c",
        "rows":rows,
    }
    (original/"factorial_source_manifest.json").write_text(json.dumps(cohort))
    def surrogate(source_path):
        assert source_path==original
        return cohort,{i:x for i,x in enumerate(observed)},{
            i:x for i,x in enumerate(rows)}
    monkeypatch.setattr(study,"_original_inputs",surrogate)
    return original


@pytest.fixture
def prepared(monkeypatch,tmp_path):
    original=_synthetic_hls_source(monkeypatch,tmp_path)
    output=tmp_path/"new_physical"
    summary=study.prepare(study=original,output=output)
    return output,summary,original


def test_prespecified_9_cases_7_controlled_pairs_and_clock():
    assert study.ORIGINAL_CASE_INDICES==(0,1,2,8,16,17,18,20,24)
    assert len(study.PAIRS)==7
    assert study.PAIRS==(
        ("storage",0,8),("binding",0,2),("padding",0,1),
        ("storage",16,24),("address",16,20),
        ("binding",16,18),("padding",16,17),
    )
    assert study.EXPECTED_CLOCK_NS==8.0
    assert "ROM_READ_ATTESTATION_PROTOCOL_V1" in study.make_vivado_tcl(
        clock_ns=study.EXPECTED_CLOCK_NS)
    assert "DRC_WARNING_OBSERVATION_PROTOCOL_V1" in study.make_vivado_tcl(
        clock_ns=study.EXPECTED_CLOCK_NS)


def test_source_preparation_and_read_only_audit_without_native(prepared):
    root,created,old=prepared
    assert created["prepared_physical_cases"]==9
    assert created["planned_matched_pairs"]==7
    assert created["original_HLS_invocations_rerun"]==0
    assert created["original_vivado_calls"]==0
    cases=study.audit(study=root)
    assert cases["original_physical_cases"]==9
    assert cases["original_planned_pairs"]==7
    assert cases["original_vivado_invocation_receipts"]==0
    assert cases["strict_clean_DRC_E4_cases"]==0
    assert all(c["status"]=="PREPARED_NOT_RUN" for c in cases["cases"])
    assert all(c["original_source_SHA_attested"] for c in cases["cases"])
    assert all(c["physical_resources"] is None for c in cases["cases"])
    assert all(c["routed_WNS_ns"] is None for c in cases["cases"])
    paired=study.summarize_pairs(cases)
    assert paired["prespecified_original_cases"]==9
    assert paired["prespecified_matched_pairs"]==7
    assert paired["strict_clean_DRC_E4_pairs"]==0
    assert paired["warning_only_descriptive_pairs"]==0
    assert paired["unknown_or_rejected_pairs"]==7
    assert all(x["physical_BRAM_delta_E4"] is None for x in paired["pairs"])
    assert all(x["pair_status"]=="UNKNOWN_NOT_BOTH_STRICT_E4"
               for x in paired["pairs"])


def test_dry_run_without_native_or_confirmation(prepared):
    root,_,_=prepared
    before={str(p.relative_to(root)):p.read_bytes()
            for p in root.rglob("*") if p.is_file()}
    result=study.execute(study=root,start_case=3,max_cases=2)
    assert result["status"]=="DRY_RUN_NO_VIVADO"
    assert result["cases"]==["case_008","case_016"]
    assert result["new_vivado_calls"]==0
    assert before=={str(p.relative_to(root)):p.read_bytes()
                   for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match="explicit confirmation"):
        study.execute(
            study=root,execute_native=True,
            confirm_isolated_native=False)
    with pytest.raises(ValueError,match="bounded to 9"):
        study.execute(study=root,start_case=8,max_cases=2)
    with pytest.raises(ValueError,match="bounded to 9"):
        study.execute(study=root,start_case=True,max_cases=1)


def test_original_rtl_tamper_refused_before_physical_run(prepared):
    root,_,_=prepared
    name=root/"case_020"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/"fpgai_model.v"
    name.write_text(name.read_text()+"\n// changed after staging\n")
    with pytest.raises(ValueError,match="RTL/ROM/Vivado Tcl altered"):
        study.audit(study=root)


def test_original_model_tamper_and_tcl_tamper_refused(prepared):
    root,_,_=prepared
    file=root/"case_016"/"model.cpp"
    file.write_text(file.read_text()+" // changed")
    with pytest.raises(ValueError,match="package altered"):
        study.audit(study=root)


def test_create_only_physical_package_and_report(prepared,tmp_path):
    root,_,original=prepared
    with pytest.raises(FileExistsError,match="create-only"):
        study.prepare(study=original,output=root)
    output=tmp_path/"read_only_report"
    before={str(p.relative_to(root)):p.read_bytes()
            for p in root.rglob("*") if p.is_file()}
    result=study.export(study=root,output=output)
    assert result["prespecified_original_cases"]==9
    assert result["prespecified_matched_pairs"]==7
    assert result["strict_clean_DRC_E4_pairs"]==0
    with pytest.raises(FileExistsError,match="create-only"):
        study.export(study=root,output=output)
    assert before=={str(p.relative_to(root)):p.read_bytes()
                   for p in root.rglob("*") if p.is_file()}


def test_strict_clean_DRC_gate_and_warning_only_remain_separate():
    case_base={
        "original_HLS_estimated_BRAM_18K":3,
        "strict_clean_DRC_E4":False,
        "routed_observation_valid":False,
        "physical_resources":None,"routed_WNS_ns":None,
    }
    rows=[]
    for i in study.ORIGINAL_CASE_INDICES:
        rows.append({
            **case_base,"case":f"case_{i:03d}",
            "graph_index":i//16,
        })
    data={"schema":study.SCHEMA,"cases":rows}
    baseline=next(x for x in rows if x["case"]=="case_016")
    storage=next(x for x in rows if x["case"]=="case_024")
    baseline.update({
        "strict_clean_DRC_E4":True,"routed_observation_valid":True,
        "physical_resources":{"bram":2.0,"lut":100,"ff":200,"dsp":4},
        "routed_WNS_ns":0.2,
    })
    storage.update({
        "strict_clean_DRC_E4":True,"routed_observation_valid":True,
        "original_HLS_estimated_BRAM_18K":2,
        "physical_resources":{"bram":1.5,"lut":102,"ff":195,"dsp":4},
        "routed_WNS_ns":0.4,
    })
    reported=study.summarize_pairs(data)
    physical=next(x for x in reported["pairs"]
                  if x["control"]=="case_016" and x["factor"]=="storage")
    assert physical["strict_clean_DRC_E4_pair"] is True
    assert physical["physical_BRAM_delta_E4"]==-0.5
    assert physical["HLS_estimated_BRAM_18K_delta"]==-1
    assert physical["routed_WNS_delta_E4_ns"]==pytest.approx(0.2)
    storage["strict_clean_DRC_E4"]=False
    descriptive=study.summarize_pairs(data)
    item=next(x for x in descriptive["pairs"]
              if x["control"]=="case_016" and x["factor"]=="storage")
    assert item["physical_BRAM_delta_E4"] is None
    assert item["routed_WNS_delta_E4_ns"] is None
    assert item["warning_only_physical_BRAM_delta"]==-0.5
    assert item["pair_status"]=="DESCRIPTIVE_ONLY_DRC_DISCLOSED"
