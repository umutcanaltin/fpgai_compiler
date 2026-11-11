"""R4h tests: original-signed source crosswalk, not mapped BRAM evidence.

Temporary synthetic RTL signatures exercise only the lexical analysis.
Vendor calls and original R4e user files are never touched.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4h_hls_memory_source_crosswalk as sut

ACC = """module fpgai_model_acc_RAM_AUTO_1R1W();
parameter DataWidth = 32;
reg [31:0] ram [0:15];
endmodule
"""
FULL = """module fpgai_model_scratch_RAM_AUTO_1R1W();
parameter DataDepth = 256;
reg [31:0] ram [0:255];
endmodule
"""
ROW = """module fpgai_model_scratch_RAM_AUTO_1R1W();
parameter DataDepth = 16;
reg [31:0] ram [0:15];
endmodule
"""
SYMBOLIC = """module scratch_RAM();
parameter DataDepth = 256;
reg [DataWidth-1:0] ram [0:DataDepth-1];
endmodule
"""
MEMORY_TYPES = ("fpgai_model_acc_RAM_AUTO_1R1W",
                "fpgai_model_scratch_RAM_AUTO_1R1W")


def _arm(root, fmt, arm, *, known=True, scratch=FULL, romname="w0.dat"):
    bits, integer = fmt
    path = (root / f"format_{bits:02d}_{integer:02d}"
            / arm / "vitis_source" / "fpgai_hls_project"
            / "solution1" / "syn" / "verilog")
    source_hashes = {}
    if known:
        path.mkdir(parents=True)
        for name, code in zip(MEMORY_TYPES, (ACC, scratch)):
            file = path / (name + ".v")
            file.write_text(code)
            source_hashes[file.name] = sha256(file.read_bytes()).hexdigest()
    return {
        "format": list(fmt), "arm": arm,
        "rtl_observation_status": (
            "ATTESTED_STATIC_TYPE_GRAPH_ONLY" if known
            else "UNKNOWN_E3_NOT_QUALIFIED"),
        "rtl_file_sha256": source_hashes if known else None,
        "rom_file_sha256": {romname: "f" * 64} if known else None,
        "provisionally_reachable_memory_module_types": (
            list(MEMORY_TYPES) if known else None),
    }


def _study(tmp_path, monkeypatch):
    root = tmp_path / "original_r4d"
    old = tmp_path / "original_r4a"
    old.mkdir()
    records = [
        _arm(root, (16, 6), "full_H", known=False),
        _arm(root, (16, 6), "row_H", known=False),
        _arm(root, (32, 6), "full_H", scratch=FULL),
        _arm(root, (32, 6), "row_H", scratch=ROW,
             romname="relocated_w0.dat"),
        _arm(root, (32, 8), "full_H", known=False),
        _arm(root, (32, 8), "row_H", known=False),
    ]
    base = {
        "prespecified_pairs": 3,
        "original_R4a_source_revision": "a" * 40,
        "original_R4d_source_revision": "b" * 40,
        "arms": records,
        "pairs": [
            {"HLS_estimated_BRAM18K_delta_row_minus_full": None},
            {"HLS_estimated_BRAM18K_delta_row_minus_full": 0},
            {"HLS_estimated_BRAM18K_delta_row_minus_full": None},
        ],
    }
    monkeypatch.setattr(sut.r4g, "inspect", lambda **kwargs: base)
    return root, old, records, base


def test_literal_reg_arrays_only_not_mapped_capacity():
    r = sut.literal_reg_arrays("module X(); reg [31:0] ram [0:255]; endmodule")
    assert r == [{
        "name": "ram", "declared_width_bits": 32,
        "declared_elements": 256, "lexical_bits_product": 8192,
        "scope": "SOURCE_DECLARATION_ONLY_NOT_MAPPED_CAPACITY",
    }]
    assert sut.literal_reg_arrays(SYMBOLIC) == []
    diag = sut.memory_source_signatures(SYMBOLIC)
    assert diag["literal_reg_array_recognizer_complete"] is False
    assert any("DataDepth" in entry["text"] for entry in
               diag["candidate_source_lines"])
    assert diag["interpretation"] == (
        "LEXICAL_SOURCE_ONLY_SYMBOLIC_DIMENSIONS_UNRESOLVED")


def test_matched_memory_names_distinguish_changed_scratch_unchanged_acc(
    tmp_path, monkeypatch,
):
    root, old, _, _ = _study(tmp_path, monkeypatch)
    result = sut.inspect(study=root, original_study=old)
    assert result["prespecified_format_pairs"] == 3
    assert result["source_crosswalk_complete_pairs"] == 1
    assert result["new_vendor_calls"] == 0
    assert result["physical_E4"] is None
    assert len(result["arms"]) == 6
    assert result["pairs"][0]["original_HLS_estimated_BRAM18K_delta"] is None
    pair = result["pairs"][1]
    assert pair["original_HLS_estimated_BRAM18K_delta"] == 0
    assert pair["original_memory_source_status"] == (
        "COMPLETE_NAMED_SOURCE_CROSSWALK_ONLY")
    assert pair["candidate_memory_module_count_full"] == 2
    assert pair["candidate_memory_module_count_row"] == 2
    assert pair["same_name_module_sources_unchanged"] == [MEMORY_TYPES[0]]
    assert pair["same_name_module_sources_changed"] == [MEMORY_TYPES[1]]
    assert pair["ROM_initialization_contents_identical_multiset"] is True
    assert pair["physical_BRAM_delta"] is None
    assert result["pairs"][2]["candidate_memory_module_count_full"] is None


def test_name_without_independent_module_file_is_partial_unknown(
    tmp_path, monkeypatch,
):
    root, old, records, _ = _study(tmp_path, monkeypatch)
    records[2]["provisionally_reachable_memory_module_types"].append(
        "missing_RAM_module")
    out = sut.inspect(study=root, original_study=old)
    assert out["source_crosswalk_complete_pairs"] == 0
    full = out["arms"][2]["memory_modules"]
    assert full[-1]["source_status"] == "UNKNOWN_NO_SAME_NAMED_SOURCE_FILE"
    assert out["pairs"][1]["same_name_module_sources_changed"] is None
    assert out["pairs"][1]["ROM_initialization_contents_identical_multiset"] is None


def test_original_hdl_sha_tamper_is_rejected(tmp_path, monkeypatch):
    root, old, _, _ = _study(tmp_path, monkeypatch)
    path = (root / "format_32_06" / "full_H" / "vitis_source"
            / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
            / (MEMORY_TYPES[1] + ".v"))
    path.write_bytes(path.read_bytes() + b"// modified after receipt\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        sut.inspect(study=root, original_study=old)


def test_original_data_immutable_and_create_only_export(tmp_path, monkeypatch):
    root, old, _, _ = _study(tmp_path, monkeypatch)
    old_sha = {
        str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }
    out = tmp_path / "new_R4h_report"
    report = sut.run(study=root, original_study=old, output=out)
    assert report["source_crosswalk_complete_pairs"] == 1
    saved = (out / "report.json").read_bytes()
    inv = json.loads((out / "artifact_sha256.json").read_text())
    assert inv["report.json"] == sha256(saved).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=root, original_study=old, output=out)
    with pytest.raises(ValueError, match="outside original"):
        sut.run(study=root, original_study=old,
                output=root / "forbidden_report")
    assert old_sha == {
        str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }
