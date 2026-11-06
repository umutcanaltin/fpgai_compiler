"""R4i parameter-default guards: synthetic source excerpts, no vendor tools.

Reported 214/134 scratch and unchanged accumulator/ROM profiles are
reproductions of the user's R4h output as tests of the pure evaluator.
They are NOT new HLS or physical observations.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4i_rtl_memory_parameter_defaults as sut

SCRATCH = "fpgai_model_scratch_RAM_AUTO_1R1W"
ACC = "fpgai_model_acc_RAM_AUTO_1R1W"
ROM_NAMES = ("fpgai_model_a_ROM_AUTO_1R", "fpgai_model_b_ROM_AUTO_1R",
             "fpgai_model_c_ROM_AUTO_1R")


def module(name, width, address_width, address_range,
           *, bad_array=False, truncate=False):
    lines = [
        f"module {name}();",
        f"parameter DataWidth = {width};",
        f"parameter AddressWidth = {address_width};",
        f"parameter AddressRange = {address_range};",
        ("reg [DataWidth-1:0] unknown[0:OtherDepth-1];" if bad_array
         else "(* ram_style = \"auto\" *)reg [DataWidth-1:0] ram[0:AddressRange-1];"),
    ]
    return {
        "module_type": name,
        "source_sha256": "b" * 64,
        "source_status": "SOURCE_SHA_ATTESTED_TEXT_ONLY",
        "signature": {
            "candidate_source_lines": [
                {"line": i + 1, "text": text}
                for i, text in enumerate(lines)],
            "source_line_snippets_truncated": truncate,
        },
    }


def arm(fmt, name, *, scratch_depth=214, acc_depth=256, known=True):
    return {
        "format": list(fmt),
        "arm": name,
        "memory_modules": None if not known else [
            module(ACC, 32, 8, acc_depth),
            module(ROM_NAMES[0], 30, 6, 64),
            module(ROM_NAMES[1], 29, 7, 128),
            module(ROM_NAMES[2], 30, 7, 128),
            module(SCRATCH, 32, 8, scratch_depth),
        ],
    }


def test_32_6_original_default_scratch_depth_is_smaller_bram_tied():
    full = sut.summarize_arm(arm((32, 6), "full_H", scratch_depth=214))
    row = sut.summarize_arm(arm((32, 6), "row_H", scratch_depth=134))
    paired = sut.compare_pair(
        full, row, format=[32, 6],
        source_bram_delta=0, rom_payloads_identical=True)
    assert paired["status"] == "SOURCE_MODULE_DEFAULT_DIMENSIONS_ONLY"
    assert paired["scratch_default_full"] == {
        "DataWidth": 32, "AddressWidth": 8, "AddressRange": 214}
    assert paired["scratch_default_row"] == {
        "DataWidth": 32, "AddressWidth": 8, "AddressRange": 134}
    assert paired["scratch_default_depth_entries_delta"] == -80
    assert paired["scratch_default_declaration_data_bits_delta"] == -2560
    assert paired["scratch_default_depth_fraction_saved"] == pytest.approx(
        80/214)
    assert paired["scratch_default_address_width_same"] is True
    assert paired["accumulator_default_unchanged"] is True
    assert paired["accumulator_default_full"]["AddressRange"] == 256
    assert paired["ROM_default_parameter_profiles_identical_multiset"] is True
    assert paired["HLS_estimated_BRAM18K_delta"] == 0
    assert paired["source_parameter_overrides_checked"] is False
    assert paired["elaborated_instance_capacity_delta"] is None
    assert paired["physical_BRAM_delta"] is None


def test_literal_width_depth_math_witness_is_not_mapped_capacity():
    result = sut.parameter_defaults(module(SCRATCH, 32, 8, 214))
    assert result["status"] == "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY"
    assert result["default_declaration_data_bits_product"] == 214*32
    assert result["default_AddressWidth_bits"] == 8
    assert result["source_parameter_line_numbers"] == {
        "DataWidth": 2, "AddressWidth": 3, "AddressRange": 4}
    assert result["assumes_instance_defaults_unoverridden"] is None
    assert result["mapped_BRAM18K"] is None


@pytest.mark.parametrize("mutation", [
    lambda d: d["signature"]["candidate_source_lines"][1].update(
        text="parameter DataWidth = SYMBOLIC_W;"),
    lambda d: d["signature"].update(source_line_snippets_truncated=True),
    lambda d: d["signature"]["candidate_source_lines"][3].update(
        text="parameter AddressRange = 1<<8;"),
    lambda d: d["signature"]["candidate_source_lines"][4].update(
        text="reg [DataWidth-1:0] ram[0:other_count-1];"),
    lambda d: d["signature"]["candidate_source_lines"].append(
        {"line": 9, "text": "parameter DataWidth = 32;"}),
    lambda d: d.update(source_status="UNKNOWN_NO_SAME_NAMED_SOURCE_FILE"),
])
def test_symbolic_ambiguous_truncated_or_unattested_is_unknown(mutation):
    m = module(SCRATCH, 32, 8, 214)
    mutation(m)
    r = sut.parameter_defaults(m)
    assert r["status"] == "UNKNOWN_SOURCE_PARAMETERS"
    assert r["default_declaration_data_bits_product"] is None
    assert r["elaborated_instance_depth"] is None
    assert r["mapped_BRAM18K"] is None


def test_impossible_source_address_range_not_extrapolated():
    r = sut.parameter_defaults(module(SCRATCH, 32, 7, 214))
    assert r["status"] == "UNKNOWN_SOURCE_PARAMETERS"
    assert r["default_AddressRange_entries"] is None


def test_full_pair_unknown_without_either_attested_native_arm():
    full = sut.summarize_arm(arm((32, 8), "full_H", known=False))
    row = sut.summarize_arm(arm((32, 8), "row_H", known=False))
    pair = sut.compare_pair(
        full, row, format=[32, 8], source_bram_delta=None,
        rom_payloads_identical=None)
    assert pair["status"] == "UNKNOWN_PARAMETER_DEFAULTS"
    assert pair["scratch_default_depth_entries_delta"] is None
    assert pair["HLS_estimated_BRAM18K_delta"] is None


def test_mismatched_arms_rejected():
    full = sut.summarize_arm(arm((32, 6), "row_H"))
    row = sut.summarize_arm(arm((32, 6), "full_H"))
    with pytest.raises(ValueError, match="mismatched"):
        sut.compare_pair(full, row, format=[32, 6],
                         source_bram_delta=0, rom_payloads_identical=True)


def test_exact_three_format_population_and_create_only(monkeypatch, tmp_path):
    old = tmp_path / "original"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    source = {
        "prespecified_format_pairs": 3,
        "original_R4a_source_revision": "a" * 40,
        "original_R4d_source_revision": "b" * 40,
        "arms": [
            arm((16, 6), "full_H", known=False),
            arm((16, 6), "row_H", known=False),
            arm((32, 6), "full_H", scratch_depth=214),
            arm((32, 6), "row_H", scratch_depth=134),
            arm((32, 8), "full_H", known=False),
            arm((32, 8), "row_H", known=False),
        ],
        "pairs": [
            {"original_HLS_estimated_BRAM18K_delta": None,
             "ROM_initialization_contents_identical_multiset": None},
            {"original_HLS_estimated_BRAM18K_delta": 0,
             "ROM_initialization_contents_identical_multiset": True},
            {"original_HLS_estimated_BRAM18K_delta": None,
             "ROM_initialization_contents_identical_multiset": None},
        ],
    }
    monkeypatch.setattr(sut.r4h, "inspect", lambda **kwargs: source)
    result = sut.inspect(study=new, original_study=old)
    assert result["prespecified_format_pairs"] == 3
    assert result["source_default_pairs_interpretable"] == 1
    assert result["pairs"][1]["scratch_default_depth_entries_delta"] == -80
    assert result["pairs"][0]["scratch_default_depth_entries_delta"] is None
    assert result["pairs"][2]["scratch_default_depth_entries_delta"] is None
    assert result["native_vendor_calls_during_analysis"] == 0
    assert result["physical_E4"] is None
    target = tmp_path / "r4i_report"
    sut.run(study=new, original_study=old, output=target)
    data = (target / "report.json").read_bytes()
    inventory = json.loads((target / "artifact_sha256.json").read_text())
    assert inventory["report.json"] == sha256(data).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=new, original_study=old, output=target)
    with pytest.raises(ValueError, match="outside original"):
        sut.run(study=new, original_study=old,
                output=new / "nested_invalid")


def test_missing_full_cohort_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(sut.r4h, "inspect",
                        lambda **_: {"prespecified_format_pairs": 3,
                                     "arms": [], "pairs": []})
    with pytest.raises(ValueError, match="incomplete"):
        sut.inspect(study=tmp_path/"x", original_study=tmp_path/"y")
