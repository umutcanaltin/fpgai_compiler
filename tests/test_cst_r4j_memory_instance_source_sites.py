"""R4j tests: source-site parsing, UNKNOWN constraints and no native calls.

Synthetic Verilog below tests the lexical parser only. It is not new
Vitis RTL or evidence that actual hardware elaborated the parameter sites.
"""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import cst_r4j_memory_instance_source_sites as sut

SCRATCH = "fpgai_model_scratch_RAM_AUTO_1R1W"
ACC = "fpgai_model_acc_RAM_AUTO_1R1W"


def rtl_text(*, depth=214, override=None, comments=False, defparam=False):
    tail = (
        f"#(.DataWidth(32), .AddressWidth(8), .AddressRange({override}))"
        if isinstance(override, int)
        else override if isinstance(override, str) else ""
    )
    return f"""module fpgai_model(input clk);
{SCRATCH} {tail} scratch_U (.clk(clk));
{ACC} acc_U (.clk(clk));
{"defparam scratch_U.AddressRange = 300;" if defparam else ""}
{"// " + SCRATCH + " fake(.clk(clk));" if comments else ""}
endmodule
module {SCRATCH}(input clk);
parameter DataWidth=32;
parameter AddressWidth=8;
parameter AddressRange={depth};
reg [DataWidth-1:0] ram[0:AddressRange-1];
endmodule
module {ACC}(input clk);
parameter DataWidth=32;
parameter AddressWidth=8;
parameter AddressRange=256;
reg [DataWidth-1:0] ram[0:AddressRange-1];
endmodule
"""


def source_profiles(depth):
    return {
        name: {
            "status": "LITERAL_MODULE_DEFAULTS_ARRAY_TEMPLATE_ONLY",
            "default_DataWidth_bits": 32,
            "default_AddressWidth_bits": 8,
            "default_AddressRange_entries": range_value,
        }
        for name, range_value in ((SCRATCH, depth), (ACC, 256))
    }


def _observation(arm, fmt, *, known, text=None):
    return {
        "format": list(fmt), "arm": arm,
        "rtl_observation_status": (
            "ATTESTED_STATIC_TYPE_GRAPH_ONLY" if known
            else "UNKNOWN_E3_NOT_QUALIFIED"),
        "rtl_file_sha256": ({"fpgai_model.v": sha256(text.encode()).hexdigest()}
                            if known else None),
        "provisionally_reachable_memory_module_types": (
            [SCRATCH, ACC] if known else None),
    }


def _default_arm(arm, fmt, *, known, depth):
    return {
        "format": list(fmt), "arm": arm,
        "status": ("SOURCE_PARAMETER_DEFAULTS_WITH_EXPLICIT_UNKNOWNS"
                   if known else "UNKNOWN_NO_ATTESTED_MEMORY_SOURCE"),
        "memory_default_profiles": source_profiles(depth) if known else None,
    }


def _study(tmp_path, monkeypatch, *, full_override=None, row_override=None,
           extra_source=None):
    new = tmp_path / "original_r4d"
    old = tmp_path / "original_r4a"
    new.mkdir()
    old.mkdir()
    g_arms, i_arms = [], []
    for fmt in ((16, 6), (32, 6), (32, 8)):
        for arm in ("full_H", "row_H"):
            known = fmt == (32, 6)
            depth = 214 if arm == "full_H" else 134
            override = full_override if arm == "full_H" else row_override
            source = (extra_source if extra_source is not None else
                      rtl_text(depth=depth, override=override))
            if known:
                folder = (new / "format_32_06" / arm / "vitis_source"
                          / "fpgai_hls_project" / "solution1" / "syn"
                          / "verilog")
                folder.mkdir(parents=True)
                (folder / "fpgai_model.v").write_text(source)
            g_arms.append(_observation(arm, fmt, known=known, text=source))
            i_arms.append(_default_arm(arm, fmt, known=known, depth=depth))
    base = {
        "original_R4a_source_revision": "a" * 40,
        "original_R4d_source_revision": "b" * 40,
        "prespecified_pairs": 3,
        "arms": g_arms,
        "pairs": [
            {"HLS_estimated_BRAM18K_delta_row_minus_full": None},
            {"HLS_estimated_BRAM18K_delta_row_minus_full": 0},
            {"HLS_estimated_BRAM18K_delta_row_minus_full": None},
        ],
    }
    dimensions = {
        "original_R4a_source_revision": "a" * 40,
        "original_R4d_source_revision": "b" * 40,
        "prespecified_format_pairs": 3,
        "arms": i_arms,
    }
    monkeypatch.setattr(sut.r4g, "inspect", lambda **_: base)
    monkeypatch.setattr(sut.r4i, "inspect", lambda **_: dimensions)
    return new, old


def test_source_site_parser_accepts_literal_named_and_no_override():
    code = rtl_text(override=134, comments=True)
    result = sut.scan_source_sites(
        {"fpgai_model.v": code}, [SCRATCH, ACC])
    assert result["status"] == "SOURCE_INSTANCE_SITES_PARSED_WITHOUT_ELABORATION"
    assert result["r2x_expected_memory_sites"] == 2
    assert len(result["instances"]) == 2
    scratch = next(s for s in result["instances"]
                   if s["module_type"] == SCRATCH)
    assert scratch["instance"] == "scratch_U"
    assert scratch["literal_named_overrides"] == {
        "DataWidth": 32, "AddressWidth": 8, "AddressRange": 134}
    assert scratch["parameter_site_status"] == "SOURCE_LITERAL_NAMED_OVERRIDES_ONLY"
    assert scratch["elaborated_instances"] is None
    assert scratch["physical_BRAM"] is None
    acc = next(s for s in result["instances"] if s["module_type"] == ACC)
    assert acc["parameter_site_status"] == (
        "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE")
    assert acc["literal_named_overrides"] == {}


@pytest.mark.parametrize("override", [
    "#(.DataWidth(WIDTH))",
    "#(32, 8, 134)",
    "#(.DataWidth(32),.DataWidth(16))",
    "#(.OtherParam(64))",
])
def test_symbolic_positional_duplicate_or_unknown_argument_fails_closed(override):
    report = sut.scan_source_sites(
        {"fpgai_model.v": rtl_text(override=override)}, [SCRATCH, ACC])
    assert report["status"] == "UNKNOWN_INCOMPLETE_OR_UNRESOLVED_INSTANCE_SITES"
    scratch = next(s for s in report["instances"]
                   if s["module_type"] == SCRATCH)
    assert scratch["literal_named_overrides"] is None
    assert "UNKNOWN" in scratch["parameter_site_status"]


def test_defparam_is_not_ignored_as_default():
    r = sut.scan_source_sites(
        {"fpgai_model.v": rtl_text(defparam=True)}, [SCRATCH, ACC])
    assert r["status"] == "UNKNOWN_INCOMPLETE_OR_UNRESOLVED_INSTANCE_SITES"
    assert r["source_defparam_present_anywhere"] is True


def test_missing_reachable_type_is_not_fabricated():
    r = sut.scan_source_sites({
        "fpgai_model.v": "module fpgai_model(); endmodule\n"
    }, [SCRATCH])
    assert r["status"] == "UNKNOWN_NO_R2X_MEMORY_INSTANCE_SITES"
    assert r["instances"] is None


def test_invalid_source_population_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        sut.scan_source_sites({}, [SCRATCH])
    with pytest.raises(ValueError, match="finite"):
        sut.scan_source_sites({"fpgai_model.v": rtl_text()}, [SCRATCH, SCRATCH])


def test_real_source_sha_checked_and_default_vs_override_not_conflated(
    tmp_path, monkeypatch,
):
    new, old = _study(tmp_path, monkeypatch,
                      full_override=214, row_override=214)
    report = sut.inspect(study=new, original_study=old)
    assert report["prespecified_format_pairs"] == 3
    assert report["original_RTL_source_arms"] == 2
    assert report["native_vendor_calls_during_analysis"] == 0
    assert report["physical_E4"] is None
    pair = report["pairs"][1]
    assert pair["status"] == "SOURCE_SCRATCH_SITES_PARSED_NOT_ELABORATED"
    assert pair["HLS_estimated_BRAM18K_delta"] == 0
    assert pair["scratch_literal_source_site_tuple_full"] == {
        "DataWidth": 32, "AddressWidth": 8, "AddressRange": 214}
    # Row source module default is 134, but this *synthetic* site
    # explicitly overrides it to 214. This must not be silently read
    # as the default-depth delta of -80.
    assert pair["scratch_literal_source_site_tuple_row"] == {
        "DataWidth": 32, "AddressWidth": 8, "AddressRange": 214}
    assert pair["scratch_literal_source_site_depth_delta"] == 0
    assert report["pairs"][0]["scratch_literal_source_site_depth_delta"] is None
    assert report["pairs"][2]["scratch_literal_source_site_depth_delta"] is None
    assert pair["elaborated_scratch_instance_depth_delta"] is None


def test_no_override_source_sites_preserve_default_difference(
    tmp_path, monkeypatch,
):
    new, old = _study(tmp_path, monkeypatch)
    data = sut.inspect(study=new, original_study=old)
    pair = data["pairs"][1]
    assert pair["scratch_literal_source_site_depth_delta"] == -80
    assert pair["scratch_site_parameter_modes"] == [
        "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE",
        "NO_EXPLICIT_OVERRIDE_AT_SOURCE_SITE"]
    assert pair["physical_BRAM_delta"] is None


def test_original_rtl_modified_after_sha_audit_is_rejected(
    tmp_path, monkeypatch,
):
    new, old = _study(tmp_path, monkeypatch)
    rtl = (new / "format_32_06" / "full_H" / "vitis_source"
           / "fpgai_hls_project" / "solution1" / "syn"
           / "verilog" / "fpgai_model.v")
    rtl.write_bytes(rtl.read_bytes() + b"\n// AFTER_NATIVE\n")
    with pytest.raises(ValueError, match="original RTL SHA mismatch"):
        sut.inspect(study=new, original_study=old)


def test_run_is_create_only_and_original_source_immutable(
    tmp_path, monkeypatch,
):
    new, old = _study(tmp_path, monkeypatch)
    before = {
        str(p.relative_to(new)): sha256(p.read_bytes()).hexdigest()
        for p in new.rglob("*.v")
    }
    out = tmp_path / "new_R4j_report"
    report = sut.run(study=new, original_study=old, output=out)
    assert report["prespecified_format_pairs"] == 3
    contents = (out / "report.json").read_bytes()
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert manifest["report.json"] == sha256(contents).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(study=new, original_study=old, output=out)
    with pytest.raises(ValueError, match="outside original evidence"):
        sut.run(study=new, original_study=old, output=new / "unsafe")
    after = {
        str(p.relative_to(new)): sha256(p.read_bytes()).hexdigest()
        for p in new.rglob("*.v")
    }
    assert before == after
