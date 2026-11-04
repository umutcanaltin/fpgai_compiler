"""R2w RTL/ROM provenance and lexical memory study: NO vendor calls.

All fixtures are synthetic to test read-only behavior, exact source/
CSynth/RTL hash verification, unknown handling and full denominator
arithmetic. Original Vitis measurements are never recreated by tests.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r2w_rtl_memory_forensics as sut


def _digest(payload:bytes)->str:
    return sha256(payload).hexdigest()


def _case(tmp_path,*,number:int=0,add_rpt:bool=True):
    folder=tmp_path/f"case_{number:03d}"
    hdl=folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    rpt=folder/"fpgai_hls_project"/"solution1"/"syn"/"report"
    hdl.mkdir(parents=True)
    rpt.mkdir()
    source=b"module fpgai_model_scratch_RAM (input clock);\nreg [15:0] scratch_data;\nendmodule\n"
    (hdl/"fpgai_model_scratch_RAM.v").write_bytes(source)
    (hdl/"original_weights.dat").write_bytes(b"1234")
    rpt_bytes=b"Scratch - report diagnostic only\nOther area reporting\n"
    if add_rpt:
        (rpt/"fpgai_model_csynth.rpt").write_bytes(rpt_bytes)
    case={"case":folder.name,"source_cpp_sha256":"b"*64}
    receipt={
        "schema":"fpgai-research/vendor-invocation/v1",
        "tool":"vitis_hls","inputs_stable_during_execution":True,
        "source_sha256":case["source_cpp_sha256"],
        "rtl_output_sha256":{"fpgai_model_scratch_RAM.v":_digest(source)},
        "rom_output_sha256":{"original_weights.dat":_digest(b"1234")},
        "fresh_report_sha256":{
            "csynth_rpt":_digest(rpt_bytes)
        } if add_rpt else {},
    }
    (folder/"vitis_invocation.json").write_text(json.dumps(receipt))
    return folder,case


def test_source_attested_rtl_and_rom_lexical_inventory(tmp_path):
    folder,case=_case(tmp_path)
    observed=sut.inspect_original_case(folder,case=case)
    assert observed["rtl_sha_attested"] is True
    assert observed["rtl_file_count"]==1
    assert observed["rom_file_count"]==1
    assert observed["scratch_name_rtl_file_count"]==1
    assert observed["memory_name_rtl_file_count"]==1
    assert observed["primitive_token_occurrences"]==0
    assert observed["report_scratch_line_count"]==1
    assert observed["rtl_observation_status"]=="LEXICAL_HDLSOURCE_ONLY"
    assert observed["original_rtl_sha256"]=={
        "fpgai_model_scratch_RAM.v":_digest(
            (folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/
             "fpgai_model_scratch_RAM.v").read_bytes())}


def test_modified_generated_verilog_refused_by_original_invocation_sha(tmp_path):
    folder,case=_case(tmp_path)
    path=(folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/
          "fpgai_model_scratch_RAM.v")
    path.write_bytes(path.read_bytes()+b"\n// source changed after Vitis\n")
    with pytest.raises(ValueError,match="RTL/ROM SHA"):
        sut.inspect_original_case(folder,case=case)


def test_modified_rom_refused_by_original_invocation_sha(tmp_path):
    folder,case=_case(tmp_path)
    path=(folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/
          "original_weights.dat")
    path.write_bytes(b"not the original weights")
    with pytest.raises(ValueError,match="RTL/ROM SHA"):
        sut.inspect_original_case(folder,case=case)


def test_modified_text_report_refused(tmp_path):
    folder,case=_case(tmp_path)
    path=(folder/"fpgai_hls_project"/"solution1"/"syn"/"report"/
          "fpgai_model_csynth.rpt")
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="text report SHA"):
        sut.inspect_original_case(folder,case=case)


def test_missing_native_or_rtl_is_unknown_never_zero(tmp_path):
    folder,case=_case(tmp_path)
    (folder/"vitis_invocation.json").unlink()
    observed=sut.inspect_original_case(folder,case=case)
    assert observed["rtl_observation_status"]=="UNKNOWN_NO_NATIVE_INVOCATION"
    assert observed["rtl_file_count"] is None
    assert observed["primitive_token_occurrences"] is None
    folder,case=_case(tmp_path,number=1,add_rpt=False)
    hdl=(folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog")
    for file in hdl.iterdir():
        file.unlink()
    hdl.rmdir()
    observed=sut.inspect_original_case(folder,case=case)
    assert observed["rtl_observation_status"]=="UNKNOWN_NO_GENERATED_RTL"
    assert observed["scratch_name_rtl_file_count"] is None


def test_source_provenance_or_symlink_refused(tmp_path):
    folder,case=_case(tmp_path)
    bad=deepcopy(case)
    bad["source_cpp_sha256"]="c"*64
    with pytest.raises(ValueError,match="invocation provenance"):
        sut.inspect_original_case(folder,case=bad)
    rtl=(folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/
         "fpgai_model_scratch_RAM.v")
    rtl.unlink()
    rtl.symlink_to(folder/"vitis_invocation.json")
    with pytest.raises(ValueError,match="symlink"):
        sut.inspect_original_case(folder,case=case)


def test_original_primitive_text_tokens_do_not_claim_physical_implementation(tmp_path):
    folder,case=_case(tmp_path,add_rpt=False)
    file=(folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/
          "fpgai_model_scratch_RAM.v")
    # Even if a primitive-looking name appears in comments, R2w
    # records only a textual occurrence, never a routed cell.
    file.write_text("module fpgai_model_scratch_RAM();\n// RAMB18E2\nendmodule\n")
    receipt=folder/"vitis_invocation.json"
    data=json.loads(receipt.read_text())
    data["rtl_output_sha256"][file.name]=_digest(file.read_bytes())
    receipt.write_text(json.dumps(data))
    observed=sut.inspect_original_case(folder,case=case)
    assert observed["primitive_token_occurrences"]==1
    assert observed["rtl_observation_status"]=="LEXICAL_HDLSOURCE_ONLY"
    assert "physical" not in observed["rtl_observation_status"].lower()


def test_read_only_run_stays_create_only_and_keeps_original_immutable(
        monkeypatch,tmp_path):
    folder,case=_case(tmp_path)
    # Only one synthetic original package here: mock the 32-case native
    # source-audit gate to exercise original-vendor-free artifact IO.
    original_case={
        **case,"graph_index":0,
        "storage":"disjoint","address":"table","binding":"auto",
        "padding":"exact","original_E2_qualified":True,
        "original_E3_qualified":True,
        "estimated_resources":{"bram_18k":1},
    }
    cases=[]
    for i in range(32):
        row={**original_case,"case":f"case_{i:03d}",
             "graph_index":i//16}
        cases.append(row)
    from fpgai_research.experiments import cst_r2w_rtl_memory_forensics as mod
    original={
        "original_vitis_invocation_receipts_present":32,
        "native_CSim_E2_qualified_cases":32,
        "native_CSynth_E3_qualified_cases":30,
        "cases":cases,
    }
    factor={
        "E3_qualified_conditional_factor_pairs":57,
        "E3_unknown_conditional_factor_pairs":7,
        "contrasts":[{
            "factor":"storage","graph_index":0,
            "arm_a":"case_000","arm_b":"case_001",
            "e3_both_qualified":False,
            "bram_18k_b_minus_a":None,
        } for _ in range(64)]
    }
    monkeypatch.setattr(mod,"audit_native",lambda **kwargs:original)
    monkeypatch.setattr(mod,"review",lambda *args,**kwargs:factor)
    monkeypatch.setattr(mod,"inspect_original_case",lambda folder,*,case:{
        "case":case["case"],"rtl_sha_attested":False,
        "rtl_file_count":None,"rom_file_count":None,
        "scratch_name_rtl_file_count":None,
        "memory_name_rtl_file_count":None,
        "primitive_token_occurrences":None,
        "report_scratch_line_count":None,
        "rtl_observation_status":"UNKNOWN_NO_GENERATED_RTL",
    })
    before={p.name:p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    result=mod.run(study=tmp_path,output=tmp_path/"fresh_review")
    assert result["original_R2u_cases"]==32
    assert result["original_R2v_factor_pairs"]==64
    assert result["source_attested_RTL_case_count"]==0
    assert result["conditional_pairs_with_both_RTL_sha_attested"]==0
    assert (tmp_path/"fresh_review"/"cases.csv").exists()
    assert (tmp_path/"fresh_review"/"contrasts.csv").exists()
    assert before=={p.name:p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError,match="create-only"):
        mod.run(study=tmp_path,output=tmp_path/"fresh_review")
