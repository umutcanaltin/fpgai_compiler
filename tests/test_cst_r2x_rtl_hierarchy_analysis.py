"""R2x static Verilog/module hierarchy controls, no vendor tools.

Generated HDL cases used below are SYNTHETIC and only test parsing,
source SHA checks, unknown status, and strictly read-only export.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r2x_rtl_hierarchy_analysis as sut


def _source():
    return {
        "fpgai_model.v": """
        // scratch_RAM fake_comment(.clk(clk));
        module fpgai_model (input clock);
          scratch_RAM #(.WIDTH(16)) mem0(.clock(clock));
          helper h0(.clk(clock));
          initial $display("RAMB18E2 example string only");
        endmodule
        module scratch_RAM #(parameter WIDTH=16) (input clock);
           reg [15:0] scratch [0:7];
        endmodule
        module helper(input clk);
           xpm_memory_sdpram xpm0(.clka(clk));
        endmodule
        module unreachable_ROM();
           reg [7:0] unused [0:15];
        endmodule
        """
    }


def test_static_verilog_module_types_and_reachable_named_memory():
    result=sut.parse_static_modules(_source())
    assert result["parse_status"]=="STATIC_TYPE_GRAPH_ONLY"
    assert result["declared_module_type_count"]==4
    assert result["static_local_instance_count"]==3
    assert result["top_status"]=="TOP_DEFINED_STATIC_ONLY"
    assert result["reachable_module_type_count"]==3
    assert result["reachable_scratch_module_type_count"]==1
    assert result["reachable_memory_module_type_count"]==1
    assert result["reachable_memory_module_type_instances"]==1
    assert result["candidate_reachable_primitive_instantiations"]==1
    assert result["reachable_reg_memory_candidate_count"]==1
    assert "unreachable_ROM" not in result[
        "provisionally_reachable_module_types"]
    assert result["unparsed_module_diagnostics"]==[]


def test_comments_and_strings_are_not_memory_instantiations():
    source={
        "fpgai_model.v": """
        module fpgai_model();
          // fake_RAM abandoned(.clk(clk));
          /* RAMB18E2 commented(.CLK(clk)); */
          initial $display("RAMB36E2 would_be_a_cell();");
        endmodule
        module fake_RAM(); endmodule
        """
    }
    got=sut.parse_static_modules(source)
    assert got["parse_status"]=="STATIC_TYPE_GRAPH_ONLY"
    assert got["static_local_instance_count"]==0
    assert got["reachable_module_type_count"]==1
    assert got["candidate_reachable_primitive_instantiations"]==0
    assert got["reachable_memory_module_type_count"]==0


def test_module_not_top_is_unknown_not_zero_memory():
    found=sut.parse_static_modules({
        "leaf.v":"module scratch_RAM(); endmodule\n"
    })
    assert found["parse_status"]=="UNKNOWN_TOP_MODULE_NOT_DEFINED"
    assert found["reachable_memory_module_type_count"] is None
    assert found["reachable_scratch_module_type_count"] is None
    assert found["top_status"]=="UNKNOWN_TOP_MODULE_NOT_DEFINED"


def test_duplicate_and_unparsed_module_never_report_safe_reachability():
    duplicate=sut.parse_static_modules({
        "a.v":"module fpgai_model(); endmodule",
        "b.v":"module fpgai_model(); endmodule"
    })
    assert duplicate["parse_status"]=="UNKNOWN_UNSUPPORTED_OR_AMBIGUOUS_VERILOG"
    assert duplicate["reachable_memory_module_type_count"] is None
    broken=sut.parse_static_modules({
        "fpgai_model.v":"module fpgai_model(); endmodule module \\special_bad(); endmodule"
    })
    assert broken["parse_status"]=="UNKNOWN_UNSUPPORTED_OR_AMBIGUOUS_VERILOG"
    assert broken["reachable_module_type_count"] is None


def test_parameterized_instance_identified_without_claiming_elaboration():
    source={
        "fpgai_model.v":"""
        module fpgai_model();
          RAMB18E2 #(.READ_WIDTH_A(18)) u_primitive(.CLKARDCLK(clk));
          sub #(.W(2), .P(4)) u_sub(.clock(clk));
        endmodule
        module sub(); endmodule
        """
    }
    r=sut.parse_static_modules(source)
    assert r["parse_status"]=="STATIC_TYPE_GRAPH_ONLY"
    assert r["candidate_reachable_primitive_instantiations"]==1
    assert r["reachable_module_type_count"]==2
    assert r["static_local_instance_count"]==2
    assert any("not preprocessed/elaborated RTL" in line for line in r["limits"])


def test_bad_filename_and_missing_input_refused():
    with pytest.raises(ValueError,match="bounded"):
        sut.parse_static_modules({})
    with pytest.raises(ValueError,match="invalid source filename"):
        sut.parse_static_modules({"../../evil.v":"module fpgai_model(); endmodule"})
    with pytest.raises(ValueError,match="bounded"):
        sut.parse_static_modules(_source(),top="bad invalid top")


def _original_fixture(monkeypatch,tmp_path):
    cases=[]
    for i in range(32):
        case=f"case_{i:03d}"
        folder=tmp_path/case/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        folder.mkdir(parents=True)
        code=("module fpgai_model();"
              " scratch_RAM a0(); endmodule"
              " module scratch_RAM();reg [15:0] scratch[0:7];endmodule")
        (folder/"fpgai_model.v").write_text(code)
        digest=sha256(code.encode()).hexdigest()
        cases.append({
            "case":case,
            "graph_index":i//16,
            "address":"table" if i%8<4 else "switch",
            "storage":"disjoint" if i%16<8 else "per_element_optimal",
            "binding":"auto" if i%4<2 else "ram_2p_bram",
            "padding":"exact" if i%2==0 else "pad64",
            "E3_qualified":i not in (4,12),
            "rtl_sha_attested":True,
            "original_rtl_sha256":{"fpgai_model.v":digest},
        })
    contrasts=[]
    for j in range(64):
        a=j%32;b=(j+1)%32
        contrasts.append({
            "factor":"storage",
            "graph_index":a//16,
            "arm_a":f"case_{a:03d}","arm_b":f"case_{b:03d}",
            "original_E3_both_qualified":a not in (4,12) and b not in (4,12),
            "both_RTL_sha_attested":True,
            "estimated_BRAM_18K_delta":None,
        })
    baseline={
        "original_source_revision":sut.ORIGINAL_SHA,
        "original_R2u_cases":32,"original_R2v_factor_pairs":64,
        "source_attested_RTL_case_count":32,
        "conditional_pairs_with_both_RTL_sha_attested":64,
        "original_E3_qualified_cases":30,
        "case_evidence":cases,"factor_evidence":contrasts,
    }
    monkeypatch.setattr(sut.r2w,"analyze",lambda **kwargs:baseline)
    return baseline


def test_read_only_original_export_without_native_tools(monkeypatch,tmp_path):
    _original_fixture(monkeypatch,tmp_path)
    before={str(p.relative_to(tmp_path)):p.read_bytes()
            for p in tmp_path.rglob("*.v")}
    output=tmp_path/"r2x_new_analysis"
    summary=sut.run(study=tmp_path,output=output)
    assert summary["original_RTL_SHA_attested_cases"]==32
    assert summary["original_R2v_conditional_pairs"]==64
    assert summary["static_top_reachable_case_count"]==32
    assert summary["static_top_reachable_pair_count"]==64
    assert summary["original_vendor_calls_during_analysis"]==0
    assert summary["physical_BRAM"] is None
    saved=json.loads((output/"summary.json").read_text())
    assert len(saved["cases"])==32
    assert len(saved["contrasts"])==64
    assert (output/"cases.csv").exists()
    assert (output/"contrasts.csv").exists()
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes()
                    for p in tmp_path.rglob("*.v")}
    with pytest.raises(FileExistsError,match="create-only"):
        sut.run(study=tmp_path,output=output)


def test_undetected_original_source_change_refused(monkeypatch,tmp_path):
    _original_fixture(monkeypatch,tmp_path)
    file=tmp_path/"case_003"/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/"fpgai_model.v"
    file.write_bytes(file.read_bytes()+b"// modified after tool\n")
    with pytest.raises(ValueError,match="originally attested RTL changed"):
        sut.analyze(study=tmp_path)
